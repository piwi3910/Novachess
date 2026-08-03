// Package train generates self-play training data and trains evaluation
// networks from it.
//
// The data format is the contract between the self-play workers and the
// trainer, and it is versioned and self-describing for a specific reason:
// mislabelled training data is the worst failure mode this pipeline has. A
// corrupt network is obvious and gets rejected by the gatekeeper. Data whose
// scores mean something slightly different from what the trainer assumes
// produces a network that is quietly, unfixably worse, and nothing downstream
// can tell you why.
package train

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/piwi3910/novachess/internal/chess"
)

// Magic identifies a Novachess training data file.
var Magic = [8]byte{'N', 'O', 'V', 'A', 'D', 'A', 'T', 'A'}

// FormatVersion is the current record layout. A reader refuses anything it does
// not recognize rather than interpreting old bytes under new rules.
const FormatVersion uint32 = 1

// RecordSize is the fixed size of one packed sample.
const RecordSize = 32

// FileExtension is what a data file is called. The trainer uses it to find a
// generation's batches when given a directory, which is how a Kubernetes Job
// names them: the runtime image has no shell to expand a glob with.
const FileExtension = ".novadata"

// Result is a game outcome, always from White's point of view.
type Result int8

const (
	BlackWon Result = -1
	Drawn    Result = 0
	WhiteWon Result = 1
)

// Score converts the result to the 0..1 target a trainer blends with the
// evaluation.
func (r Result) Score() float64 {
	switch r {
	case WhiteWon:
		return 1
	case BlackWon:
		return 0
	default:
		return 0.5
	}
}

// Sample is one training position.
//
// Score and Result are both from White's perspective, deliberately and
// unlike the engine's own convention, which is relative to the side to move.
// A file where some numbers are white-relative and others are mover-relative
// is impossible to interpret correctly without knowing which is which, and
// getting it wrong trains the network to prefer losing positions. One
// orientation for everything removes the question.
type Sample struct {
	// Position at the time of the search.
	Position *chess.Position
	// Score is the search evaluation in centipawns, from White's perspective.
	Score int16
	// Result is the eventual game outcome, from White's perspective.
	Result Result
	// Ply is the move number at which the position occurred, which lets the
	// trainer weight early and late positions differently.
	Ply uint16
}

// packed is the on-disk layout, exactly RecordSize bytes.
//
//	 0..7   occupancy bitboard
//	 8..23  one nibble per occupied square, in square order
//	24..25  side to move, castling rights and en passant file
//	26..27  score, from White's perspective
//	28..29  ply
//	30      result
//	31      reserved, must be zero
//
// The nibble array is what keeps a record small: a position has at most 32
// pieces, so four bits each fits in sixteen bytes, and the occupancy bitboard
// says which squares they belong to. Storing a FEN string instead would be
// roughly three times the size for a billion-position dataset.
type packed [RecordSize]byte

// EncodeSample packs a sample into its on-disk form.
//
// Chess960 positions are rejected: castling rights are stored as four bits,
// which cannot express arbitrary rook files. Self-play is classical, so this
// costs nothing and refusing is better than silently writing a position that
// decodes wrong.
func EncodeSample(s Sample) (packed, error) {
	var p packed

	if s.Position == nil {
		return p, fmt.Errorf("train: sample has no position")
	}
	if s.Position.IsChess960() {
		return p, fmt.Errorf("train: the packed format does not support Chess960 castling")
	}

	occ := s.Position.Occupied()
	if n := occ.Count(); n > 32 {
		return p, fmt.Errorf("train: %d pieces on the board, the format holds at most 32", n)
	}
	binary.LittleEndian.PutUint64(p[0:8], uint64(occ))

	// Walk the occupied squares in index order so the decoder can pair each
	// nibble with the corresponding set bit.
	index := 0
	for bb := occ; bb != 0; index++ {
		var sq chess.Square
		sq, bb = bb.PopFirst()

		piece := s.Position.PieceAt(sq)
		if piece >= chess.NoPiece {
			return p, fmt.Errorf("train: occupancy claims a piece on %v but the board is empty there", sq)
		}

		offset := 8 + index/2
		if index%2 == 0 {
			p[offset] = byte(piece)
		} else {
			p[offset] |= byte(piece) << 4
		}
	}

	var meta uint16
	if s.Position.SideToMove() == chess.Black {
		meta |= 1
	}
	meta |= uint16(s.Position.CastlingRights()) << 1
	// The en passant file, or 8 for none, in the next four bits.
	epField := uint16(8)
	if ep := s.Position.EnPassantSquare(); ep != chess.NoSquare {
		epField = uint16(ep.File())
	}
	meta |= epField << 5
	binary.LittleEndian.PutUint16(p[24:26], meta)

	binary.LittleEndian.PutUint16(p[26:28], uint16(s.Score))
	binary.LittleEndian.PutUint16(p[28:30], s.Ply)

	// The result is the one field nothing downstream can sanity-check: any byte
	// decodes to some Result, and a wrong one teaches the network the opposite
	// of what the game showed. A value outside the three outcomes means the
	// caller has a bug, and writing it would bury that bug in a dataset.
	switch s.Result {
	case WhiteWon, Drawn, BlackWon:
	default:
		return packed{}, fmt.Errorf("train: result %d is not a game outcome", s.Result)
	}
	p[30] = byte(s.Result)

	return p, nil
}

// DecodeSample unpacks a record.
//
// Every field is checked before it is used. Decoding is the side of this format
// that reads bytes it did not write — a truncated upload, a half-flushed file
// from a worker that was evicted, a length that disagrees with the header — so
// a malformed record has to come back as an error rather than as a panic in the
// middle of a training run.
func DecodeSample(p packed) (Sample, error) {
	var s Sample

	occ := chess.Bitboard(binary.LittleEndian.Uint64(p[0:8]))
	// The nibble array holds 32 pieces. A larger occupancy would read past the
	// end of the record, so it is rejected before anything indexes into it.
	if n := occ.Count(); n > 32 {
		return s, fmt.Errorf("train: occupancy claims %d pieces, the format holds at most 32", n)
	}
	if p[31] != 0 {
		return s, fmt.Errorf("train: reserved byte is %d, want 0", p[31])
	}

	meta := binary.LittleEndian.Uint16(p[24:26])

	side := chess.White
	if meta&1 != 0 {
		side = chess.Black
	}
	castling := chess.CastlingRights((meta >> 1) & 0xF)
	// The en passant field is a file, or 8 for none. Anything else would fall
	// through to "no en passant" and silently decode as a different position
	// from the one that was written.
	epFile := int((meta >> 5) & 0xF)
	if epFile > 8 {
		return s, fmt.Errorf("train: en passant field is %d, want a file or 8 for none", epFile)
	}

	// Rebuild the position through FEN rather than by poking at internals, so
	// a decoded sample goes through exactly the same validation as any other
	// position and cannot enter the pipeline in an impossible state.
	board := [chess.SquareCount]chess.Piece{}
	for i := range board {
		board[i] = chess.NoPiece
	}

	index := 0
	for bb := occ; bb != 0; index++ {
		var sq chess.Square
		sq, bb = bb.PopFirst()

		offset := 8 + index/2
		nibble := p[offset]
		if index%2 == 0 {
			nibble &= 0xF
		} else {
			nibble >>= 4
		}
		if nibble >= byte(chess.NoPiece) {
			return s, fmt.Errorf("train: invalid piece code %d on %v", nibble, sq)
		}
		board[sq] = chess.Piece(nibble)
	}

	fen := buildFEN(board, side, castling, epFile)
	pos, err := chess.ParseFEN(fen)
	if err != nil {
		return s, fmt.Errorf("train: decoded an unusable position %q: %w", fen, err)
	}

	s.Position = pos
	s.Score = int16(binary.LittleEndian.Uint16(p[26:28]))
	s.Ply = binary.LittleEndian.Uint16(p[28:30])
	s.Result = Result(int8(p[30]))
	return s, nil
}

// buildFEN renders a decoded board as a FEN string.
func buildFEN(board [chess.SquareCount]chess.Piece, side chess.Color, castling chess.CastlingRights, epFile int) string {
	var out []byte

	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			piece := board[chess.MakeSquare(file, rank)]
			if piece >= chess.NoPiece {
				empty++
				continue
			}
			if empty > 0 {
				out = append(out, byte('0'+empty))
				empty = 0
			}
			out = append(out, piece.String()[0])
		}
		if empty > 0 {
			out = append(out, byte('0'+empty))
		}
		if rank > 0 {
			out = append(out, '/')
		}
	}

	out = append(out, ' ')
	if side == chess.White {
		out = append(out, 'w')
	} else {
		out = append(out, 'b')
	}

	out = append(out, ' ')
	out = append(out, []byte(castling.String())...)

	out = append(out, ' ')
	if epFile >= 0 && epFile < 8 {
		// The rank follows from the side to move: only the player about to
		// move can capture en passant.
		rank := byte('6')
		if side == chess.Black {
			rank = '3'
		}
		out = append(out, byte('a'+epFile), rank)
	} else {
		out = append(out, '-')
	}

	// The halfmove and fullmove counters are not stored: neither affects the
	// evaluation of a position, and a billion records is not the place to
	// spend four bytes apiece on something nothing reads.
	out = append(out, ' ', '0', ' ', '1')
	return string(out)
}

// Writer appends samples to a stream.
type Writer struct {
	w       io.Writer
	written int
}

// NewWriter writes the file header and returns a writer for the records.
func NewWriter(w io.Writer) (*Writer, error) {
	var header [16]byte
	copy(header[0:8], Magic[:])
	binary.LittleEndian.PutUint32(header[8:12], FormatVersion)
	binary.LittleEndian.PutUint32(header[12:16], RecordSize)

	if _, err := w.Write(header[:]); err != nil {
		return nil, fmt.Errorf("train: write header: %w", err)
	}
	return &Writer{w: w}, nil
}

// Write appends one sample.
func (w *Writer) Write(s Sample) error {
	p, err := EncodeSample(s)
	if err != nil {
		return err
	}
	if _, err := w.w.Write(p[:]); err != nil {
		return fmt.Errorf("train: write record: %w", err)
	}
	w.written++
	return nil
}

// WriteAll appends a batch.
func (w *Writer) WriteAll(samples []Sample) error {
	for i, s := range samples {
		if err := w.Write(s); err != nil {
			return fmt.Errorf("train: sample %d: %w", i, err)
		}
	}
	return nil
}

// Count returns how many samples have been written.
func (w *Writer) Count() int { return w.written }

// Reader reads samples from a stream.
type Reader struct {
	r io.Reader
}

// NewReader validates the header and returns a reader.
//
// A wrong magic or version is a hard error rather than a best-effort read.
// Interpreting an old layout under new rules would produce plausible-looking
// positions with meaningless labels, which is exactly the failure this format
// is versioned to prevent.
func NewReader(r io.Reader) (*Reader, error) {
	var header [16]byte
	if _, err := io.ReadFull(r, header[:]); err != nil {
		return nil, fmt.Errorf("train: read header: %w", err)
	}

	if [8]byte(header[0:8]) != Magic {
		return nil, fmt.Errorf("train: not a Novachess data file")
	}
	if v := binary.LittleEndian.Uint32(header[8:12]); v != FormatVersion {
		return nil, fmt.Errorf("train: data file is format version %d, this build reads version %d", v, FormatVersion)
	}
	if size := binary.LittleEndian.Uint32(header[12:16]); size != RecordSize {
		return nil, fmt.Errorf("train: data file has %d-byte records, this build expects %d", size, RecordSize)
	}

	return &Reader{r: r}, nil
}

// Read returns the next sample, or io.EOF at the end of the stream.
func (r *Reader) Read() (Sample, error) {
	var p packed
	if _, err := io.ReadFull(r.r, p[:]); err != nil {
		if err == io.ErrUnexpectedEOF {
			return Sample{}, fmt.Errorf("train: truncated record: %w", err)
		}
		return Sample{}, err
	}
	return DecodeSample(p)
}

// ReadAll reads to the end of the stream.
func (r *Reader) ReadAll() ([]Sample, error) {
	var out []Sample
	for {
		s, err := r.Read()
		if err == io.EOF {
			return out, nil
		}
		if err != nil {
			return out, err
		}
		out = append(out, s)
	}
}
