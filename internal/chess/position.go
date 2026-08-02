package chess

import (
	"fmt"
	"strconv"
	"strings"
)

// StartingFEN is the standard chess starting position.
const StartingFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

// undo records everything MakeMove destroys that cannot be recovered from the
// move alone, so that UnmakeMove can restore the position exactly.
type undo struct {
	move     Move
	captured Piece
	castling CastlingRights
	epSquare Square
	halfmove int
	key      uint64
}

// Position is a chess position. It maintains redundant representations —
// bitboards by piece type and color, plus a square-indexed mailbox — because
// each answers a different question cheaply: bitboards for "where are all the
// knights", mailbox for "what is on e4".
//
// A Position is not safe for concurrent use. Search threads each hold their
// own; the self-play workers do the same.
type Position struct {
	byType  [PieceTypeCount]Bitboard
	byColor [ColorCount]Bitboard
	board   [SquareCount]Piece

	side     Color
	castling CastlingRights
	epSquare Square
	halfmove int // plies since the last capture or pawn move (fifty-move rule)
	fullmove int // move number, starting at 1 and incremented after Black moves

	key uint64

	history []undo
}

// castlingRightsMask[s] holds the rights that survive when square s is touched
// by a move, as either origin or destination. A king or rook leaving its home
// square loses the corresponding rights, and so does a rook being captured on
// its home square — the latter is easy to forget and produces subtle bugs that
// only perft catches.
var castlingRightsMask [SquareCount]CastlingRights

func init() {
	for i := range castlingRightsMask {
		castlingRightsMask[i] = AllCastling
	}
	castlingRightsMask[E1] &^= WhiteKingSide | WhiteQueenSide
	castlingRightsMask[H1] &^= WhiteKingSide
	castlingRightsMask[A1] &^= WhiteQueenSide
	castlingRightsMask[E8] &^= BlackKingSide | BlackQueenSide
	castlingRightsMask[H8] &^= BlackKingSide
	castlingRightsMask[A8] &^= BlackQueenSide
}

// NewPosition returns the standard starting position.
func NewPosition() *Position {
	p, err := ParseFEN(StartingFEN)
	if err != nil {
		panic("chess: starting FEN is invalid: " + err.Error())
	}
	return p
}

// emptyPosition returns a position with no pieces, ready to be filled in.
func emptyPosition() *Position {
	p := &Position{
		epSquare: NoSquare,
		fullmove: 1,
		history:  make([]undo, 0, 1024),
	}
	for i := range p.board {
		p.board[i] = NoPiece
	}
	return p
}

// SideToMove returns the color to move.
func (p *Position) SideToMove() Color { return p.side }

// CastlingRights returns the castling rights still available.
func (p *Position) CastlingRights() CastlingRights { return p.castling }

// EnPassantSquare returns the square a pawn may capture onto en passant, or
// NoSquare if the previous move was not a double pawn push.
func (p *Position) EnPassantSquare() Square { return p.epSquare }

// HalfmoveClock returns the number of plies since the last capture or pawn
// move, for the fifty-move rule.
func (p *Position) HalfmoveClock() int { return p.halfmove }

// FullmoveNumber returns the move number, starting from 1.
func (p *Position) FullmoveNumber() int { return p.fullmove }

// Key returns the Zobrist hash of the position.
func (p *Position) Key() uint64 { return p.key }

// PieceAt returns the piece on s, or NoPiece if the square is empty.
func (p *Position) PieceAt(s Square) Piece { return p.board[s] }

// Occupied returns every occupied square.
func (p *Position) Occupied() Bitboard { return p.byColor[White] | p.byColor[Black] }

// ColorPieces returns every piece of the given color.
func (p *Position) ColorPieces(c Color) Bitboard { return p.byColor[c] }

// Pieces returns every piece of the given type, of both colors.
func (p *Position) Pieces(pt PieceType) Bitboard { return p.byType[pt] }

// ColorPiecesOfType returns the pieces of a given color and type.
func (p *Position) ColorPiecesOfType(c Color, pt PieceType) Bitboard {
	return p.byColor[c] & p.byType[pt]
}

// KingSquare returns the square of the given color's king. Positions without a
// king are rejected at parse time, so this always finds one.
func (p *Position) KingSquare(c Color) Square {
	return (p.byColor[c] & p.byType[King]).First()
}

// putPiece places a piece on an empty square and folds it into the hash.
func (p *Position) putPiece(pc Piece, s Square) {
	p.board[s] = pc
	p.byType[pc.Type()] |= s.BB()
	p.byColor[pc.Color()] |= s.BB()
	p.key ^= zobristPieces[pc][s]
}

// removePiece clears an occupied square and folds the piece out of the hash.
func (p *Position) removePiece(s Square) {
	pc := p.board[s]
	p.board[s] = NoPiece
	p.byType[pc.Type()] &^= s.BB()
	p.byColor[pc.Color()] &^= s.BB()
	p.key ^= zobristPieces[pc][s]
}

// movePiece relocates a piece between two squares, which must be empty at the
// destination.
func (p *Position) movePiece(from, to Square) {
	pc := p.board[from]
	delta := from.BB() | to.BB()
	p.board[from] = NoPiece
	p.board[to] = pc
	p.byType[pc.Type()] ^= delta
	p.byColor[pc.Color()] ^= delta
	p.key ^= zobristPieces[pc][from] ^ zobristPieces[pc][to]
}

// AttackersTo returns every piece of either color attacking square s under the
// given occupancy. Occupancy is a parameter rather than read from the position
// so that callers can ask hypothetical questions — most importantly "is this
// square still attacked once the king steps off it", where the king must be
// removed from the occupancy or it would shield itself from a slider.
func (p *Position) AttackersTo(s Square, occ Bitboard) Bitboard {
	rooksQueens := p.byType[Rook] | p.byType[Queen]
	bishopsQueens := p.byType[Bishop] | p.byType[Queen]

	// Pawn attacks are inverted here: squares from which a white pawn attacks
	// s are exactly the squares a black pawn on s would attack.
	return (PawnAttacks[Black][s] & p.byColor[White] & p.byType[Pawn]) |
		(PawnAttacks[White][s] & p.byColor[Black] & p.byType[Pawn]) |
		(KnightAttacks[s] & p.byType[Knight]) |
		(KingAttacks[s] & p.byType[King]) |
		(RookAttacks(s, occ) & rooksQueens) |
		(BishopAttacks(s, occ) & bishopsQueens)
}

// IsAttacked reports whether square s is attacked by any piece of color by.
func (p *Position) IsAttacked(s Square, by Color) bool {
	return p.AttackersTo(s, p.Occupied())&p.byColor[by] != 0
}

// Checkers returns the pieces giving check to the side to move.
func (p *Position) Checkers() Bitboard {
	ksq := p.KingSquare(p.side)
	return p.AttackersTo(ksq, p.Occupied()) & p.byColor[p.side.Opposite()]
}

// InCheck reports whether the side to move is in check.
func (p *Position) InCheck() bool { return p.Checkers() != 0 }

// MakeMove applies m to the position. The move must be legal; passing an
// illegal move corrupts the position rather than returning an error, because
// this sits in the search hot path and every caller generates its moves from
// LegalMoves.
func (p *Position) MakeMove(m Move) {
	us := p.side
	them := us.Opposite()
	from, to := m.From(), m.To()
	kind := m.Kind()

	captured := NoPiece
	if kind == KindEnPassant {
		captured = MakePiece(them, Pawn)
	} else if kind != KindCastling {
		// Castling is encoded king-to-king-square, so a rook standing on the
		// destination is our own and must not be read as a capture.
		captured = p.board[to]
	}

	p.history = append(p.history, undo{
		move:     m,
		captured: captured,
		castling: p.castling,
		epSquare: p.epSquare,
		halfmove: p.halfmove,
		key:      p.key,
	})

	// Clear any existing en passant square; it is only valid for one ply.
	if p.epSquare != NoSquare {
		p.key ^= zobristEPFile[p.epSquare.File()]
		p.epSquare = NoSquare
	}

	moved := p.board[from]
	p.halfmove++

	switch kind {
	case KindCastling:
		rookFrom, rookTo := castlingRookSquares(to)
		p.movePiece(from, to)
		p.movePiece(rookFrom, rookTo)

	case KindEnPassant:
		// The captured pawn sits beside the destination, not on it.
		capSq := Square(int(to) - PawnPush(us))
		p.removePiece(capSq)
		p.movePiece(from, to)
		p.halfmove = 0

	case KindPromotion:
		if captured != NoPiece {
			p.removePiece(to)
		}
		p.removePiece(from)
		p.putPiece(MakePiece(us, m.Promotion()), to)
		p.halfmove = 0

	default:
		if captured != NoPiece {
			p.removePiece(to)
			p.halfmove = 0
		}
		p.movePiece(from, to)
		if moved.Type() == Pawn {
			p.halfmove = 0
			// A double push sets the en passant target square behind the pawn.
			if abs(int(to)-int(from)) == 16 {
				p.epSquare = Square(int(from) + PawnPush(us))
				p.key ^= zobristEPFile[p.epSquare.File()]
			}
		}
	}

	// Touching a king or rook home square revokes the matching rights, whether
	// the piece moved from it or was captured on it.
	if p.castling != NoCastling {
		newRights := p.castling & castlingRightsMask[from] & castlingRightsMask[to]
		if newRights != p.castling {
			p.key ^= zobristCastling[p.castling] ^ zobristCastling[newRights]
			p.castling = newRights
		}
	}

	if us == Black {
		p.fullmove++
	}
	p.side = them
	p.key ^= zobristSide
}

// UnmakeMove reverts the most recent MakeMove. Calling it on a position with no
// history panics, which catches unbalanced make/unmake pairs immediately rather
// than letting them corrupt a search.
func (p *Position) UnmakeMove() {
	n := len(p.history) - 1
	if n < 0 {
		panic("chess: UnmakeMove with empty history")
	}
	u := p.history[n]
	p.history = p.history[:n]

	// Restore the side first so "us" refers to the side that made the move.
	p.side = p.side.Opposite()
	us := p.side
	from, to := u.move.From(), u.move.To()

	switch u.move.Kind() {
	case KindCastling:
		rookFrom, rookTo := castlingRookSquares(to)
		p.movePiece(to, from)
		p.movePiece(rookTo, rookFrom)

	case KindEnPassant:
		p.movePiece(to, from)
		capSq := Square(int(to) - PawnPush(us))
		p.putPiece(MakePiece(us.Opposite(), Pawn), capSq)

	case KindPromotion:
		p.removePiece(to)
		p.putPiece(MakePiece(us, Pawn), from)
		if u.captured != NoPiece {
			p.putPiece(u.captured, to)
		}

	default:
		p.movePiece(to, from)
		if u.captured != NoPiece {
			p.putPiece(u.captured, to)
		}
	}

	if us == Black {
		p.fullmove--
	}
	p.castling = u.castling
	p.epSquare = u.epSquare
	p.halfmove = u.halfmove
	// The incremental XORs above have scrambled the key; the saved one is
	// authoritative and cheaper than undoing each term.
	p.key = u.key
}

// castlingRookSquares maps the king's destination square to the rook's origin
// and destination.
func castlingRookSquares(kingTo Square) (from, to Square) {
	switch kingTo {
	case G1:
		return H1, F1
	case C1:
		return A1, D1
	case G8:
		return H8, F8
	case C8:
		return A8, D8
	}
	panic("chess: invalid castling destination " + kingTo.String())
}

// ParseFEN parses a position in Forsyth-Edwards Notation. The halfmove clock
// and fullmove number are optional, defaulting to 0 and 1, since many test
// suites and opening books omit them.
func ParseFEN(fen string) (*Position, error) {
	fields := strings.Fields(fen)
	if len(fields) < 4 {
		return nil, fmt.Errorf("chess: FEN needs at least 4 fields, got %d: %q", len(fields), fen)
	}

	p := emptyPosition()

	ranks := strings.Split(fields[0], "/")
	if len(ranks) != 8 {
		return nil, fmt.Errorf("chess: FEN board needs 8 ranks, got %d", len(ranks))
	}
	// FEN lists rank 8 first, but our square numbering starts at rank 1.
	for i, row := range ranks {
		rank := 7 - i
		file := 0
		for _, ch := range []byte(row) {
			switch {
			case ch >= '1' && ch <= '8':
				file += int(ch - '0')
			default:
				idx := strings.IndexByte(pieceChars, ch)
				if idx < 0 {
					return nil, fmt.Errorf("chess: invalid piece %q in FEN", string(ch))
				}
				if file > 7 {
					return nil, fmt.Errorf("chess: rank %d overflows in FEN", rank+1)
				}
				p.putPiece(Piece(idx), MakeSquare(file, rank))
				file++
			}
		}
		if file != 8 {
			return nil, fmt.Errorf("chess: rank %d has %d files, want 8", rank+1, file)
		}
	}

	switch fields[1] {
	case "w":
		p.side = White
	case "b":
		p.side = Black
	default:
		return nil, fmt.Errorf("chess: invalid side to move %q", fields[1])
	}

	if fields[2] != "-" {
		for _, ch := range []byte(fields[2]) {
			switch ch {
			case 'K':
				p.castling |= WhiteKingSide
			case 'Q':
				p.castling |= WhiteQueenSide
			case 'k':
				p.castling |= BlackKingSide
			case 'q':
				p.castling |= BlackQueenSide
			default:
				return nil, fmt.Errorf("chess: invalid castling right %q", string(ch))
			}
		}
	}

	if fields[3] != "-" {
		sq, ok := parseSquare(fields[3])
		if !ok {
			return nil, fmt.Errorf("chess: invalid en passant square %q", fields[3])
		}
		p.epSquare = sq
	}

	if len(fields) >= 5 {
		n, err := strconv.Atoi(fields[4])
		if err != nil {
			return nil, fmt.Errorf("chess: invalid halfmove clock %q", fields[4])
		}
		p.halfmove = n
	}
	if len(fields) >= 6 {
		n, err := strconv.Atoi(fields[5])
		if err != nil {
			return nil, fmt.Errorf("chess: invalid fullmove number %q", fields[5])
		}
		p.fullmove = n
	}

	if err := p.validate(); err != nil {
		return nil, err
	}

	p.key = p.computeKey()
	return p, nil
}

// validate rejects positions the move generator would misbehave on. It is
// deliberately not a full legality check — it only enforces the invariants the
// rest of the package relies on.
func (p *Position) validate() error {
	for c := White; c <= Black; c++ {
		kings := p.ColorPiecesOfType(c, King)
		if kings.Count() != 1 {
			return fmt.Errorf("chess: %s has %d kings, want exactly 1", c, kings.Count())
		}
	}
	// The side that just moved must not be left in check.
	if p.IsAttacked(p.KingSquare(p.side.Opposite()), p.side) {
		return fmt.Errorf("chess: %s is in check but it is %s to move", p.side.Opposite(), p.side)
	}
	if p.byType[Pawn]&(RankBB[0]|RankBB[7]) != 0 {
		return fmt.Errorf("chess: pawns on the first or eighth rank")
	}
	return nil
}

// FEN renders the position in Forsyth-Edwards Notation.
func (p *Position) FEN() string {
	var sb strings.Builder

	for rank := 7; rank >= 0; rank-- {
		empty := 0
		for file := 0; file < 8; file++ {
			pc := p.board[MakeSquare(file, rank)]
			if pc == NoPiece {
				empty++
				continue
			}
			if empty > 0 {
				sb.WriteString(strconv.Itoa(empty))
				empty = 0
			}
			sb.WriteByte(pieceChars[pc])
		}
		if empty > 0 {
			sb.WriteString(strconv.Itoa(empty))
		}
		if rank > 0 {
			sb.WriteByte('/')
		}
	}

	if p.side == White {
		sb.WriteString(" w ")
	} else {
		sb.WriteString(" b ")
	}
	sb.WriteString(p.castling.String())
	sb.WriteByte(' ')
	sb.WriteString(p.epSquare.String())
	sb.WriteByte(' ')
	sb.WriteString(strconv.Itoa(p.halfmove))
	sb.WriteByte(' ')
	sb.WriteString(strconv.Itoa(p.fullmove))

	return sb.String()
}

// String renders the board as ASCII art with the FEN beneath it, for debugging.
func (p *Position) String() string {
	var sb strings.Builder
	for rank := 7; rank >= 0; rank-- {
		sb.WriteString(strconv.Itoa(rank + 1))
		sb.WriteByte(' ')
		for file := 0; file < 8; file++ {
			sb.WriteString(p.board[MakeSquare(file, rank)].String())
			sb.WriteByte(' ')
		}
		sb.WriteByte('\n')
	}
	sb.WriteString("  a b c d e f g h\n")
	sb.WriteString(p.FEN())
	sb.WriteByte('\n')
	return sb.String()
}

// Clone returns an independent copy of the position, sharing no state. Search
// threads use this to get their own root position.
func (p *Position) Clone() *Position {
	c := *p
	c.history = make([]undo, len(p.history), max(cap(p.history), 1024))
	copy(c.history, p.history)
	return &c
}
