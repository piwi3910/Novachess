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

	// chess960 selects how castling moves and the FEN castling field are
	// written. Move generation does not consult it: the generator is
	// generalized and correct for both variants.
	chess960 bool

	// castlingRook holds each right's rook origin, and castlingPath the
	// squares that must be empty for it. Both are per-position because
	// Chess960 starts the rooks on arbitrary files.
	castlingRook [castlingIndexCount]Square
	castlingPath [castlingIndexCount]Bitboard

	// crMask[s] holds the rights that survive when square s is touched by a
	// move, as either origin or destination. A king or rook leaving its home
	// square loses the corresponding rights, and so does a rook captured on
	// its home square — the latter is easy to forget and produces bugs only
	// perft catches. This is per-position for the same reason as the above.
	crMask [SquareCount]CastlingRights

	history []undo
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
	for i := range p.crMask {
		p.crMask[i] = AllCastling
	}
	for i := range p.castlingRook {
		p.castlingRook[i] = NoSquare
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
		// The move is encoded king-takes-rook, so `to` is the rook's square.
		// Both pieces are lifted before either is placed: in Chess960 their
		// origins and destinations overlap freely — a king may not move at
		// all, and king and rook may swap squares — so any move-then-move
		// order would overwrite a piece that has not been read yet.
		idx := castlingIndex(us, isKingSideCastling(m))
		p.removePiece(from)
		p.removePiece(to)
		p.putPiece(MakePiece(us, King), castlingKingTo(idx))
		p.putPiece(MakePiece(us, Rook), castlingRookTo(idx))

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
			// A double push sets the en passant target square behind the pawn,
			// but only when an opponent pawn can actually capture there.
			//
			// This is a rules requirement, not an optimization. FIDE compares
			// positions for repetition by the moves available in them, so two
			// positions differing only by an en passant target that no pawn
			// could ever use are the same position and must hash identically.
			// Recording the square unconditionally would make a threefold
			// repetition go undetected, losing a draw the rules grant.
			if abs(int(to)-int(from)) == 16 {
				if epSq := Square(int(from) + PawnPush(us)); p.epCaptureAvailable(epSq, them) {
					p.epSquare = epSq
					p.key ^= zobristEPFile[epSq.File()]
				}
			}
		}
	}

	// Touching a king or rook home square revokes the matching rights, whether
	// the piece moved from it or was captured on it.
	if p.castling != NoCastling {
		newRights := p.castling & p.crMask[from] & p.crMask[to]
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
		// Mirror of MakeMove: lift both pieces from their destinations before
		// restoring either origin, since the four squares may overlap.
		idx := castlingIndex(us, isKingSideCastling(u.move))
		p.removePiece(castlingKingTo(idx))
		p.removePiece(castlingRookTo(idx))
		p.putPiece(MakePiece(us, King), from)
		p.putPiece(MakePiece(us, Rook), to)

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

// MakeNullMove passes the turn without moving a piece.
//
// This is not a legal chess move. It exists for null-move pruning, where the
// search asks "if I did nothing at all, could my opponent still not hurt me?"
// and prunes the subtree when the answer is no. It must never be played in a
// position where the side to move is in check, since the resulting position
// would have a capturable king, and the search must not use it in endgames
// where zugzwang makes doing nothing genuinely better than moving.
//
// Must be paired with UnmakeNullMove, not UnmakeMove.
func (p *Position) MakeNullMove() {
	p.history = append(p.history, undo{
		move:     MoveNone,
		captured: NoPiece,
		castling: p.castling,
		epSquare: p.epSquare,
		halfmove: p.halfmove,
		key:      p.key,
	})

	if p.epSquare != NoSquare {
		p.key ^= zobristEPFile[p.epSquare.File()]
		p.epSquare = NoSquare
	}

	// Reset the halfmove clock so repetition detection cannot look back past
	// this point. Positions on either side of a null move are not reachable
	// from one another by legal play, so treating them as repetitions would
	// invent draws that do not exist.
	p.halfmove = 0

	if p.side == Black {
		p.fullmove++
	}
	p.side = p.side.Opposite()
	p.key ^= zobristSide
}

// UnmakeNullMove reverts MakeNullMove.
func (p *Position) UnmakeNullMove() {
	n := len(p.history) - 1
	if n < 0 {
		panic("chess: UnmakeNullMove with empty history")
	}
	u := p.history[n]
	if !u.move.IsNone() {
		panic("chess: UnmakeNullMove on a real move")
	}
	p.history = p.history[:n]

	p.side = p.side.Opposite()
	if p.side == Black {
		p.fullmove--
	}
	p.castling = u.castling
	p.epSquare = u.epSquare
	p.halfmove = u.halfmove
	p.key = u.key
}

// IsCapture reports whether a move captures an enemy piece, including en
// passant. Move ordering leans on this heavily, so it avoids generating
// anything.
func (p *Position) IsCapture(m Move) bool {
	return m.Kind() == KindEnPassant || (m.Kind() != KindCastling && p.board[m.To()] != NoPiece)
}

// epCaptureAvailable reports whether the given side has a pawn that can legally
// capture en passant onto epSq.
//
// "Legally" is doing real work here: a pawn may attack the square and still be
// unable to make the capture, because en passant removes two pieces from one
// rank at once and can expose its own king. Such a position has no en passant
// possibility at all, and must not record one.
func (p *Position) epCaptureAvailable(epSq Square, capturer Color) bool {
	if epSq >= NoSquare || p.board[epSq] != NoPiece {
		return false
	}
	them := capturer.Opposite()

	// The pawn to be captured must actually be standing beside the target.
	capSq := Square(int(epSq) - PawnPush(capturer))
	if p.board[capSq] != MakePiece(them, Pawn) {
		return false
	}

	// Squares from which a pawn of the capturing color attacks epSq are the
	// squares a pawn of the opposite color standing on epSq would attack.
	candidates := PawnAttacks[them][epSq] & p.ColorPiecesOfType(capturer, Pawn)
	if candidates == 0 {
		return false
	}

	ksq := p.KingSquare(capturer)
	occ := p.Occupied()
	for candidates != 0 {
		var from Square
		from, candidates = candidates.PopFirst()
		if p.enPassantLegal(from, epSq, capturer, ksq, occ) {
			return true
		}
	}
	return false
}

// ParseFEN parses a position in Forsyth-Edwards Notation. The halfmove clock
// and fullmove number are optional, defaulting to 0 and 1, since many test
// suites and opening books omit them.
func ParseFEN(fen string) (*Position, error) { return parseFEN(fen, false) }

// ParseFENVariant parses a FEN, forcing the Chess960 interpretation on or off
// rather than inferring it.
//
// Inference works only when the castling field uses Shredder notation, so a
// Chess960 position that has already lost both castling rights is
// indistinguishable from a classical one — and harmlessly so, since the flag
// only affects how castling is written and there is no castling left to write.
// UCI callers should use this with the value of the UCI_Chess960 option rather
// than relying on inference.
func ParseFENVariant(fen string, chess960 bool) (*Position, error) { return parseFEN(fen, chess960) }

func parseFEN(fen string, chess960 bool) (*Position, error) {
	fields := strings.Fields(fen)
	if len(fields) < 4 {
		return nil, fmt.Errorf("chess: FEN needs at least 4 fields, got %d: %q", len(fields), fen)
	}

	p := emptyPosition()
	p.chess960 = chess960

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

	// Castling needs the kings located, so reject kingless positions before
	// resolving it rather than letting KingSquare read an empty bitboard.
	for c := White; c <= Black; c++ {
		if n := p.ColorPiecesOfType(c, King).Count(); n != 1 {
			return nil, fmt.Errorf("chess: %s has %d kings, want exactly 1", c, n)
		}
	}

	if err := p.parseCastlingField(fields[2]); err != nil {
		return nil, err
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
	p.sanitize()

	p.key = p.computeKey()
	return p, nil
}

// parseCastlingField reads the castling field of a FEN, in either the classical
// KQkq form or the Shredder form that names each rook by file (AHah).
//
// A right whose rook cannot be located is silently dropped. That is the
// sanitization path for castling: believing an unsupported right is a crash,
// because the generator would emit a castling move and playing it would
// relocate a piece that is not there.
func (p *Position) parseCastlingField(field string) error {
	if field == "-" {
		return nil
	}

	for _, ch := range []byte(field) {
		var us Color
		var rookFrom Square
		var kingSide bool

		switch {
		case ch == 'K', ch == 'Q', ch == 'k', ch == 'q':
			us = White
			if ch == 'k' || ch == 'q' {
				us = Black
			}
			kingSide = ch == 'K' || ch == 'k'

			found, ok := p.findCastlingRook(us, kingSide)
			if !ok {
				continue // no rook to castle with; drop the right
			}
			rookFrom = found

		case ch >= 'A' && ch <= 'H', ch >= 'a' && ch <= 'h':
			// Shredder notation names the rook's file directly, which is the
			// only unambiguous form when both rooks sit on the same side of
			// the king or the king is not on the e-file.
			var rank, file int
			if ch >= 'a' {
				us, rank, file = Black, 7, int(ch-'a')
			} else {
				us, rank, file = White, 0, int(ch-'A')
			}
			rookFrom = MakeSquare(file, rank)
			if p.board[rookFrom] != MakePiece(us, Rook) {
				continue // drop a right whose rook is not there
			}
			ksq := p.KingSquare(us)
			if ksq.Rank() != rank {
				continue // a king off its back rank cannot castle
			}
			kingSide = rookFrom > ksq
			p.chess960 = true

		default:
			return fmt.Errorf("chess: invalid castling right %q", string(ch))
		}

		p.setCastlingRight(castlingIndex(us, kingSide), rookFrom)
	}
	return nil
}

// sanitize corrects fields that a FEN may assert but the board contradicts.
//
// These are silently repaired rather than rejected, because FENs arrive from
// GUIs, opening books and web APIs that are routinely sloppy about them, and
// refusing an otherwise perfectly legal position over a stale castling flag
// would break interoperability for no benefit. What must not happen is
// believing them.
func (p *Position) sanitize() {
	// Castling rights are already sanitized during parsing: a right is only
	// recorded once its rook has been located on the board, so an unsupported
	// flag never reaches this point.

	// An en passant target no pawn can legally use is not part of the
	// position: FIDE distinguishes positions by the moves available in them.
	// Keeping it would hash this position differently from the identical one
	// reached by another move order, hiding a repetition draw.
	if p.epSquare != NoSquare && !p.epCaptureAvailable(p.epSquare, p.side) {
		p.epSquare = NoSquare
	}
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
	sb.WriteString(p.castlingFEN())
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
