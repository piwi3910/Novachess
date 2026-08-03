package eval

import "github.com/piwi3910/novachess/internal/chess"

// HCE is a hand-crafted evaluation.
//
// It is deliberately readable rather than maximally strong. Its job is to get
// the engine playing well enough to generate the first generation of self-play
// data; NNUE trained on that data will replace it. Terms are kept separable so
// the weights can be Texel-tuned against self-play results before that happens.
//
// HCE holds no state and is safe for concurrent use, which matters because
// every lazy-SMP search thread shares one.
type HCE struct{}

// NewHCE returns a hand-crafted evaluator.
func NewHCE() *HCE { return &HCE{} }

// Tunable weights. These are hand-chosen starting points, not tuned values.
const (
	bishopPairBonus   = 30
	tempoBonus        = 10
	isolatedPawnPenal = 15
	doubledPawnPenal  = 10
	rookOpenFile      = 25
	rookSemiOpenFile  = 12
)

// Passed pawns are scored by how far they have travelled, counted from their
// own side's back rank. The endgame values are much larger because a passer's
// whole point is queening, and in the middlegame it is usually still
// blockadable by pieces that will not be there later.
var (
	mgPassedByRank = [8]int{0, 5, 10, 20, 35, 60, 100, 0}
	egPassedByRank = [8]int{0, 10, 20, 40, 70, 120, 180, 0}
)

// Mobility is scored per safe square reached, weighted by piece type. Knights
// gain little from an extra square; rooks and queens gain a lot.
var (
	mgMobility = [chess.PieceTypeCount]int{0, 4, 5, 2, 1, 0}
	egMobility = [chess.PieceTypeCount]int{0, 4, 5, 4, 2, 0}
)

// kingAttackWeight scores how dangerous each attacking piece type is near the
// enemy king.
var kingAttackWeight = [chess.PieceTypeCount]int{0, 20, 20, 40, 80, 0}

// Game phase weights. A full board scores 24; a bare king-and-pawn endgame
// scores 0. Everything tapers between the two.
var phaseWeight = [chess.PieceTypeCount]int{0, 1, 1, 2, 4, 0}

const maxPhase = 24

// Evaluate scores the position from the perspective of the side to move.
func (e *HCE) Evaluate(p *chess.Position) int {
	// Terms are accumulated for White and for the middlegame and endgame
	// separately, then interpolated once at the end. Interpolating each term as
	// it is computed would be both slower and harder to reason about.
	var mg, eg, phase int

	for c := chess.White; c <= chess.Black; c++ {
		cmg, ceg, cphase := e.evaluateSide(p, c)
		phase += cphase
		if c == chess.White {
			mg += cmg
			eg += ceg
		} else {
			mg -= cmg
			eg -= ceg
		}
	}

	// Phase can exceed the nominal maximum after promotions produce more
	// material than the starting position has.
	if phase > maxPhase {
		phase = maxPhase
	}
	score := (mg*phase + eg*(maxPhase-phase)) / maxPhase

	// The right to move is worth something in itself, and giving it a value
	// keeps the evaluation asymmetric in the way real chess is.
	if p.SideToMove() == chess.White {
		score += tempoBonus
	} else {
		score -= tempoBonus
	}

	if p.SideToMove() == chess.Black {
		score = -score
	}
	return score
}

// evaluateSide accumulates every term for one color, returning middlegame and
// endgame scores plus that side's contribution to the game phase.
func (e *HCE) evaluateSide(p *chess.Position, c chess.Color) (mg, eg, phase int) {
	them := c.Opposite()
	occ := p.Occupied()
	ourPawns := p.ColorPiecesOfType(c, chess.Pawn)
	theirPawns := p.ColorPiecesOfType(them, chess.Pawn)

	// Squares attacked by enemy pawns are not mobility: a knight can move
	// there, but it will simply be taken.
	theirPawnAttacks := pawnAttacks(theirPawns, them)

	theirKingRing := kingRing[p.KingSquare(them)]
	kingAttackScore := 0

	for pt := chess.Pawn; pt <= chess.King; pt++ {
		for bb := p.ColorPiecesOfType(c, pt); bb != 0; {
			var s chess.Square
			s, bb = bb.PopFirst()

			phase += phaseWeight[pt]
			mg += mgMaterial[pt] + pst(mgTable[pt], c, s)
			eg += egMaterial[pt] + pst(egTable[pt], c, s)

			if pt == chess.Pawn || pt == chess.King {
				continue
			}

			// Mobility, and pressure on the enemy king, share one attack set.
			attacks := chess.AttacksFrom(pt, c, s, occ)
			safe := attacks &^ theirPawnAttacks &^ p.ColorPieces(c)
			n := safe.Count()
			mg += mgMobility[pt] * n
			eg += egMobility[pt] * n

			if ring := attacks & theirKingRing; ring != 0 {
				kingAttackScore += kingAttackWeight[pt] * ring.Count()
			}

			if pt == chess.Rook {
				file := chess.FileBB[s.File()]
				switch {
				case ourPawns&file == 0 && theirPawns&file == 0:
					mg += rookOpenFile
					eg += rookOpenFile / 2
				case ourPawns&file == 0:
					mg += rookSemiOpenFile
					eg += rookSemiOpenFile / 2
				}
			}
		}
	}

	// Two bishops cover both color complexes, which is worth more than the sum
	// of the individual pieces.
	if p.ColorPiecesOfType(c, chess.Bishop).Count() >= 2 {
		mg += bishopPairBonus
		eg += bishopPairBonus
	}

	pmg, peg := pawnStructure(ourPawns, theirPawns, c)
	mg += pmg
	eg += peg

	// King attacks matter in the middlegame and are nearly irrelevant once the
	// heavy pieces come off, so the term is deliberately not added to eg.
	mg += kingAttackScore / 4

	return mg, eg, phase
}

// pawnStructure scores isolated, doubled and passed pawns.
func pawnStructure(ourPawns, theirPawns chess.Bitboard, c chess.Color) (mg, eg int) {
	for bb := ourPawns; bb != 0; {
		var s chess.Square
		s, bb = bb.PopFirst()

		file := s.File()

		// Isolated: no friendly pawn on either adjacent file, so it can never
		// be defended by a pawn.
		if ourPawns&adjacentFiles[file] == 0 {
			mg -= isolatedPawnPenal
			eg -= isolatedPawnPenal
		}

		// Doubled: another friendly pawn ahead on the same file. Counting only
		// forward avoids penalising the same pair twice.
		if ourPawns&forwardFile[c][s] != 0 {
			mg -= doubledPawnPenal
			eg -= doubledPawnPenal
		}

		// Passed: no enemy pawn ahead on this or an adjacent file.
		if theirPawns&passedMask[c][s] == 0 {
			// Rank counted from the pawn's own side, so a White pawn on rank 7
			// and a Black pawn on rank 2 score identically.
			rank := s.Rank()
			if c == chess.Black {
				rank = 7 - rank
			}
			mg += mgPassedByRank[rank]
			eg += egPassedByRank[rank]
		}
	}
	return mg, eg
}

// pawnAttacks returns every square attacked by the given pawns.
func pawnAttacks(pawns chess.Bitboard, c chess.Color) chess.Bitboard {
	if c == chess.White {
		return pawns.Shifted(chess.North+chess.East) | pawns.Shifted(chess.North+chess.West)
	}
	return pawns.Shifted(chess.South+chess.East) | pawns.Shifted(chess.South+chess.West)
}
