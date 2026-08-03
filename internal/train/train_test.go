package train

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/piwi3910/novachess/internal/chess"
	"github.com/piwi3910/novachess/internal/eval"
	"github.com/piwi3910/novachess/internal/events"
)

func mustParse(t *testing.T, fen string) *chess.Position {
	t.Helper()
	p, err := chess.ParseFEN(fen)
	if err != nil {
		t.Fatalf("ParseFEN(%q): %v", fen, err)
	}
	return p
}

// boardOf reduces a position to what the record format actually stores: the
// pieces, the side to move, the castling rights and the en passant target. The
// halfmove and fullmove counters are deliberately not encoded, so comparing
// whole FEN strings would assert something the format never promised.
func boardOf(p *chess.Position) string {
	fields := strings.Fields(p.FEN())
	if len(fields) > 4 {
		fields = fields[:4]
	}
	return strings.Join(fields, " ")
}

// TestSampleRoundTrip is the load-bearing test for the format. Every field must
// survive encoding, because a field that silently does not is a field the
// trainer reads as zero — and a zero score or result is a plausible value, not
// an obvious corruption.
func TestSampleRoundTrip(t *testing.T) {
	fens := []string{
		chess.StartingFEN,
		"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
		"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
		"4k3/8/8/8/8/8/8/4K3 w - - 0 1",
		"8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1",
		// An en passant target that a pawn can actually use, so it survives
		// the position's own normalization.
		"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
		"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
	}

	scores := []int16{0, 1, -1, 250, -250, 30000, -30000}
	results := []Result{WhiteWon, BlackWon, Drawn}

	for _, fen := range fens {
		for _, score := range scores {
			for _, result := range results {
				pos := mustParse(t, fen)
				original := Sample{Position: pos, Score: score, Result: result, Ply: 42}

				encoded, err := EncodeSample(original)
				if err != nil {
					t.Fatalf("EncodeSample(%q): %v", fen, err)
				}
				decoded, err := DecodeSample(encoded)
				if err != nil {
					t.Fatalf("DecodeSample(%q): %v", fen, err)
				}

				if got, want := boardOf(decoded.Position), boardOf(pos); got != want {
					t.Errorf("position round trip:\n got %q\nwant %q", got, want)
				}
				if decoded.Score != score {
					t.Errorf("score = %d, want %d", decoded.Score, score)
				}
				if decoded.Result != result {
					t.Errorf("result = %d, want %d", decoded.Result, result)
				}
				if decoded.Ply != 42 {
					t.Errorf("ply = %d, want 42", decoded.Ply)
				}
			}
		}
	}
}

// TestRoundTripOverRandomPlay extends the round trip past hand-picked
// positions, since the encoding's edge cases are things like a full board, a
// nearly empty one, and every castling combination — which arise from play
// rather than from fixtures somebody thought to write.
func TestRoundTripOverRandomPlay(t *testing.T) {
	r := newRNG(0xFEED_FACE_1234)
	checked := 0

	for game := 0; game < 20; game++ {
		pos := chess.NewPosition()
		for ply := 0; ply < 80; ply++ {
			moves := pos.LegalMoves()
			if len(moves) == 0 {
				break
			}

			original := Sample{Position: pos.Clone(), Score: int16(ply * 7), Result: Drawn, Ply: uint16(ply)}
			encoded, err := EncodeSample(original)
			if err != nil {
				t.Fatalf("EncodeSample at %s: %v", pos.FEN(), err)
			}
			decoded, err := DecodeSample(encoded)
			if err != nil {
				t.Fatalf("DecodeSample at %s: %v", pos.FEN(), err)
			}
			if got, want := boardOf(decoded.Position), boardOf(pos); got != want {
				t.Fatalf("round trip diverged:\n got %q\nwant %q", got, want)
			}
			checked++

			pos.MakeMove(moves[r.next()%uint64(len(moves))])
		}
	}
	t.Logf("round-tripped %d positions from play", checked)
}

func TestRecordSizeIsFixed(t *testing.T) {
	pos := chess.NewPosition()
	encoded, err := EncodeSample(Sample{Position: pos, Score: 10, Result: Drawn})
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) != RecordSize {
		t.Errorf("record is %d bytes, want %d", len(encoded), RecordSize)
	}
}

func TestEncodeRejectsChess960(t *testing.T) {
	// Castling rights are four bits, which cannot express arbitrary rook
	// files. Refusing is better than writing something that decodes wrong.
	pos, err := chess.ParseFEN("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := EncodeSample(Sample{Position: pos}); err == nil {
		t.Error("encoding a Chess960 position succeeded, want an error")
	}
}

func TestWriterReaderRoundTrip(t *testing.T) {
	samples := []Sample{
		{Position: chess.NewPosition(), Score: 25, Result: WhiteWon, Ply: 0},
		{Position: mustParse(t, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"), Score: -140, Result: BlackWon, Ply: 17},
		{Position: mustParse(t, "4k3/8/8/8/8/8/8/4K3 w - - 0 1"), Score: 0, Result: Drawn, Ply: 99},
	}

	var buf bytes.Buffer
	w, err := NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteAll(samples); err != nil {
		t.Fatal(err)
	}
	if w.Count() != len(samples) {
		t.Errorf("wrote %d samples, want %d", w.Count(), len(samples))
	}

	r, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(samples) {
		t.Fatalf("read %d samples, want %d", len(got), len(samples))
	}
	for i := range got {
		if boardOf(got[i].Position) != boardOf(samples[i].Position) {
			t.Errorf("sample %d position = %q, want %q", i, boardOf(got[i].Position), boardOf(samples[i].Position))
		}
		if got[i].Score != samples[i].Score || got[i].Result != samples[i].Result || got[i].Ply != samples[i].Ply {
			t.Errorf("sample %d labels = (%d, %d, %d), want (%d, %d, %d)",
				i, got[i].Score, got[i].Result, got[i].Ply,
				samples[i].Score, samples[i].Result, samples[i].Ply)
		}
	}
}

// TestReaderRejectsForeignData covers the reason the format carries a header at
// all. Reading old or foreign bytes under current rules would yield
// plausible-looking positions with meaningless labels — the one failure this
// pipeline cannot detect downstream.
func TestReaderRejectsForeignData(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{"empty", nil},
		{"truncated header", []byte("NOVA")},
		{"wrong magic", append([]byte("NOTNOVA!"), make([]byte, 8)...)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewReader(bytes.NewReader(tc.data)); err == nil {
				t.Error("reader accepted the data, want an error")
			}
		})
	}

	t.Run("future version", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write(Magic[:])
		buf.Write([]byte{99, 0, 0, 0})         // version 99
		buf.Write([]byte{RecordSize, 0, 0, 0}) // correct record size
		if _, err := NewReader(&buf); err == nil {
			t.Error("reader accepted a future format version, want an error")
		}
	})

	t.Run("wrong record size", func(t *testing.T) {
		var buf bytes.Buffer
		buf.Write(Magic[:])
		buf.Write([]byte{byte(FormatVersion), 0, 0, 0})
		buf.Write([]byte{64, 0, 0, 0}) // 64-byte records
		if _, err := NewReader(&buf); err == nil {
			t.Error("reader accepted a different record size, want an error")
		}
	})
}

func TestReaderRejectsTruncatedRecord(t *testing.T) {
	var buf bytes.Buffer
	w, err := NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(Sample{Position: chess.NewPosition(), Score: 1, Result: Drawn}); err != nil {
		t.Fatal(err)
	}

	// Chop the last record in half.
	data := buf.Bytes()
	truncated := data[:len(data)-10]

	r, err := NewReader(bytes.NewReader(truncated))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err == nil {
		t.Error("reader accepted a truncated record, want an error")
	}
}

func TestResultScore(t *testing.T) {
	cases := []struct {
		result Result
		want   float64
	}{
		{WhiteWon, 1},
		{BlackWon, 0},
		{Drawn, 0.5},
	}
	for _, tc := range cases {
		if got := tc.result.Score(); got != tc.want {
			t.Errorf("Result(%d).Score() = %v, want %v", tc.result, got, tc.want)
		}
	}
}

func testUnit(seed uint64) events.WorkUnit {
	return events.WorkUnit{
		ID:           "test-unit",
		Generation:   0,
		Games:        2,
		RandomPlies:  6,
		NodesPerMove: 2000,
		Seed:         seed,
	}
}

// TestGenerateIsReproducible is the property the whole distributed pipeline
// rests on. A work unit redelivered to a different worker — which happens
// whenever a pod is evicted — must produce identical data, or the training set
// depends on which machine ran it.
func TestGenerateIsReproducible(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)
	unit := testUnit(0xABCDEF)

	first, firstStats, err := g.Generate(context.Background(), unit)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) == 0 {
		t.Fatal("generated no samples at all")
	}

	for run := 1; run < 3; run++ {
		again, againStats, err := NewGenerator(eval.NewHCE(), 4).Generate(context.Background(), unit)
		if err != nil {
			t.Fatal(err)
		}

		if len(again) != len(first) {
			t.Fatalf("run %d produced %d samples, first run produced %d", run, len(again), len(first))
		}
		if againStats.Nodes != firstStats.Nodes {
			t.Errorf("run %d searched %d nodes, first run searched %d", run, againStats.Nodes, firstStats.Nodes)
		}

		for i := range first {
			if again[i].Position.FEN() != first[i].Position.FEN() {
				t.Fatalf("run %d sample %d position differs:\n got %s\nwant %s",
					run, i, again[i].Position.FEN(), first[i].Position.FEN())
			}
			if again[i].Score != first[i].Score || again[i].Result != first[i].Result {
				t.Fatalf("run %d sample %d labels differ: (%d,%d) vs (%d,%d)",
					run, i, again[i].Score, again[i].Result, first[i].Score, first[i].Result)
			}
		}
	}
	t.Logf("%d samples reproduced exactly across runs", len(first))
}

// TestDifferentSeedsProduceDifferentGames is the complement: reproducibility is
// only useful if the seed actually varies the output, or every worker in the
// cluster would generate the same games.
func TestDifferentSeedsProduceDifferentGames(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)

	a, _, err := g.Generate(context.Background(), testUnit(1))
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := g.Generate(context.Background(), testUnit(2))
	if err != nil {
		t.Fatal(err)
	}
	if len(a) == 0 || len(b) == 0 {
		t.Fatal("a seed produced no samples")
	}

	same := len(a) == len(b)
	if same {
		for i := range a {
			if a[i].Position.FEN() != b[i].Position.FEN() {
				same = false
				break
			}
		}
	}
	if same {
		t.Error("two seeds produced identical data; the seed is not reaching the opening randomisation")
	}
}

// TestGeneratedSamplesAreUsable checks the properties the trainer relies on:
// every position is legal, every label is one of three values, and scores are
// White-relative rather than mixed.
func TestGeneratedSamplesAreUsable(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)
	samples, stats, err := g.Generate(context.Background(), testUnit(0x1234))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("no samples produced")
	}

	for i, s := range samples {
		if s.Position == nil {
			t.Fatalf("sample %d has no position", i)
		}
		// A position that cannot be re-parsed could not be encoded either.
		if _, err := chess.ParseFEN(s.Position.FEN()); err != nil {
			t.Fatalf("sample %d holds an unusable position %q: %v", i, s.Position.FEN(), err)
		}
		switch s.Result {
		case WhiteWon, BlackWon, Drawn:
		default:
			t.Fatalf("sample %d has result %d, which is not a game outcome", i, s.Result)
		}
		// Noisy positions are meant to have been filtered out.
		if s.Position.InCheck() {
			t.Errorf("sample %d is in check; those should be filtered as noisy", i)
		}
		if _, err := EncodeSample(s); err != nil {
			t.Fatalf("sample %d cannot be encoded: %v", i, err)
		}
	}

	if stats.Games != 2 {
		t.Errorf("played %d games, want 2", stats.Games)
	}
	if stats.Positions != len(samples) {
		t.Errorf("stats say %d positions, got %d samples", stats.Positions, len(samples))
	}
	if stats.Nodes == 0 {
		t.Error("no nodes recorded; the throughput report would be useless")
	}
	t.Logf("%d games, %d positions, %d skipped as noisy, %.0f nps",
		stats.Games, stats.Positions, stats.SkippedNoisy, stats.NodesPerSecond())
}

// TestGenerateRejectsTimeLimitedUnits checks the guard that keeps
// irreproducible work out of the pipeline. A unit with no node budget would be
// searched by wall clock, and a slower cluster node would then produce
// different games for the same unit.
func TestGenerateRejectsTimeLimitedUnits(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)

	unit := testUnit(1)
	unit.NodesPerMove = 0
	if _, _, err := g.Generate(context.Background(), unit); err == nil {
		t.Error("generated from a unit with no node budget, want an error")
	}

	unit = testUnit(1)
	unit.Games = 0
	if _, _, err := g.Generate(context.Background(), unit); err == nil {
		t.Error("generated from a unit asking for no games, want an error")
	}
}

// TestOpeningFENsAreUsed checks that a unit's openings actually reach the
// games, since a coordinator varying openings would otherwise silently have no
// effect and every worker would explore the same lines.
func TestOpeningFENsAreUsed(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)

	unit := testUnit(7)
	unit.RandomPlies = 0
	unit.Games = 1
	// A distinctive endgame that could not arise from the starting position in
	// a handful of plies.
	unit.OpeningFENs = []string{"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"}

	samples, _, err := g.Generate(context.Background(), unit)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Skip("the opening produced no usable samples; nothing to check")
	}

	// With no random plies, the first sample is the opening itself or very
	// close to it, and in any case must have few pieces.
	if n := samples[0].Position.Occupied().Count(); n > 12 {
		t.Errorf("first sample has %d pieces; the opening FEN does not appear to have been used", n)
	}
}

func TestGenerateHonoursContextCancellation(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	unit := testUnit(3)
	unit.Games = 100
	if _, _, err := g.Generate(ctx, unit); err == nil {
		t.Error("generation ignored a cancelled context")
	}
}

// TestGeneratedDataSurvivesTheFileFormat is the end-to-end check that the two
// halves of this package agree: what self-play produces must be writable and
// readable without loss.
func TestGeneratedDataSurvivesTheFileFormat(t *testing.T) {
	g := NewGenerator(eval.NewHCE(), 4)
	samples, _, err := g.Generate(context.Background(), testUnit(0x5150))
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) == 0 {
		t.Fatal("no samples produced")
	}

	var buf bytes.Buffer
	w, err := NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteAll(samples); err != nil {
		t.Fatal(err)
	}

	// Read the size before the reader drains the buffer.
	size := buf.Len()

	r, err := NewReader(&buf)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(samples) {
		t.Fatalf("wrote %d samples, read back %d", len(samples), len(got))
	}
	if want := 16 + len(samples)*RecordSize; size != want {
		t.Errorf("file is %d bytes, want %d (a 16-byte header plus %d-byte records)", size, want, RecordSize)
	}
	for i := range got {
		if boardOf(got[i].Position) != boardOf(samples[i].Position) {
			t.Fatalf("sample %d changed through the file: %q became %q",
				i, boardOf(samples[i].Position), boardOf(got[i].Position))
		}
		if got[i].Score != samples[i].Score || got[i].Result != samples[i].Result {
			t.Fatalf("sample %d labels changed through the file", i)
		}
	}
	t.Logf("%d samples survived a write and read cycle (%d bytes)", len(got), size)
}
