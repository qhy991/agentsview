package parser

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// kwMessage is a minimal KernelOwl transcript message used by the
// test fixtures.
type kwMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	Timestamp string         `json:"timestamp"`
	Usage     map[string]int `json:"usage,omitempty"`
}

// kwWriteConversation writes a conversation_conv_*.json transcript
// into dir. taskID populates metadata.task_id (empty for
// experiment-level conversations).
func kwWriteConversation(
	t *testing.T, dir, name, taskID string, msgs []kwMessage,
) {
	t.Helper()
	doc := map[string]any{
		"conversation_id": name,
		"created_at":      "2026-03-19T10:35:17.980467",
		"updated_at":      "2026-03-19T10:36:00.000000",
		"messages":        msgs,
	}
	if taskID != "" {
		doc["metadata"] = map[string]string{"task_id": taskID}
	}
	data, err := json.Marshal(doc)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(
		filepath.Join(dir, name), data, 0o644,
	))
}

// kwWriteMeta writes a session_metadata.json file.
func kwWriteMeta(t *testing.T, dir, id, kind string) {
	t.Helper()
	doc := map[string]string{
		"session_id":   id,
		"session_kind": kind,
		"created_at":   "2026-05-23T10:24:50Z",
		"updated_at":   "2026-05-23T10:30:00Z",
	}
	data, err := json.Marshal(doc)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(
		filepath.Join(dir, "session_metadata.json"), data, 0o644,
	))
}

func TestParseKernelOwlTime(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want time.Time
	}{
		{"empty", "", time.Time{}},
		{"rfc3339Z", "2026-05-23T10:24:50Z",
			time.Date(2026, 5, 23, 10, 24, 50, 0, time.UTC)},
		{"naive_micros", "2026-03-19T10:35:17.980467",
			time.Date(2026, 3, 19, 10, 35, 17, 980467000, time.UTC)},
		{"naive_seconds", "2026-03-19T10:35:17",
			time.Date(2026, 3, 19, 10, 35, 17, 0, time.UTC)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseKernelOwlTime(tt.in)
			if tt.want.IsZero() {
				assert.True(t, got.IsZero())
				return
			}
			assert.True(t, got.Equal(tt.want),
				"got %v want %v", got, tt.want)
		})
	}
}

func TestParseKernelOwlMessage(t *testing.T) {
	t.Run("assistant usage maps to context/output tokens", func(t *testing.T) {
		raw := `{"role":"assistant","content":"hi","timestamp":"2026-03-19T10:35:17.980467","usage":{"input_tokens":240,"output_tokens":1523,"cache_read_input_tokens":10,"cache_creation_input_tokens":5}}`
		m, ok := parseKernelOwlMessage(jsonResult(t, raw))
		require.True(t, ok)
		assert.Equal(t, RoleAssistant, m.Role)
		assert.False(t, m.IsSystem)
		// context = input + cache_creation + cache_read
		assert.Equal(t, 255, m.ContextTokens)
		assert.Equal(t, 1523, m.OutputTokens)
		assert.True(t, m.HasContextTokens)
		assert.True(t, m.HasOutputTokens)
	})
	t.Run("system role is flagged", func(t *testing.T) {
		raw := `{"role":"system","content":"you are helpful","timestamp":"2026-03-19T10:35:17"}`
		m, ok := parseKernelOwlMessage(jsonResult(t, raw))
		require.True(t, ok)
		assert.True(t, m.IsSystem)
	})
	t.Run("empty non-system content is dropped", func(t *testing.T) {
		raw := `{"role":"assistant","content":"   ","timestamp":"2026-03-19T10:35:17"}`
		_, ok := parseKernelOwlMessage(jsonResult(t, raw))
		assert.False(t, ok)
	})
}

func TestParseKernelOwlExperimentSingleRunFoldsIntoParent(t *testing.T) {
	dir := t.TempDir()
	expDir := filepath.Join(dir, "autoresearch_20260530_211801_abc")
	require.NoError(t, os.MkdirAll(expDir, 0o755))
	kwWriteMeta(t, expDir, "autoresearch_20260530_211801_abc", "autoresearch")
	kwWriteConversation(t, expDir, "conversation_conv_1.json", "run_001_main",
		[]kwMessage{
			{Role: "user", Content: "what is tma", Timestamp: "2026-03-19T10:35:17", Usage: map[string]int{"input_tokens": 10}},
			{Role: "assistant", Content: "tensor memory accelerator", Timestamp: "2026-03-19T10:35:20", Usage: map[string]int{"output_tokens": 100}},
		})

	results, err := ParseKernelOwlExperiment(expDir, "", "local")
	require.NoError(t, err)
	require.Len(t, results, 1)

	sess := results[0].Session
	assert.Equal(t, "kernelowl:autoresearch_20260530_211801_abc", sess.ID)
	assert.Empty(t, sess.ParentSessionID)
	assert.Equal(t, RelNone, sess.RelationshipType)
	assert.Equal(t, AgentKernelOwl, sess.Agent)
	assert.Equal(t, "autoresearch", sess.Project)
	assert.Len(t, results[0].Messages, 2)
	assert.Equal(t, 2, sess.MessageCount)
	assert.Equal(t, 1, sess.UserMessageCount)
	assert.Equal(t, "what is tma", sess.FirstMessage)
	assert.Equal(t, 100, sess.TotalOutputTokens)
}

func TestParseKernelOwlExperimentMultiRunEmitsSubagents(t *testing.T) {
	dir := t.TempDir()
	expDir := filepath.Join(dir, "autoresearch_20260601_101801_def")
	require.NoError(t, os.MkdirAll(expDir, 0o755))
	kwWriteMeta(t, expDir, "autoresearch_20260601_101801_def", "autoresearch")
	// Two runs with distinct task_ids -> multi-run.
	kwWriteConversation(t, expDir, "conversation_conv_a.json", "run_alpha",
		[]kwMessage{{Role: "user", Content: "alpha q", Timestamp: "2026-06-01T10:00:00"}})
	kwWriteConversation(t, expDir, "conversation_conv_b.json", "run_beta",
		[]kwMessage{{Role: "user", Content: "beta q", Timestamp: "2026-06-01T10:01:00"}})

	results, err := ParseKernelOwlExperiment(expDir, "", "local")
	require.NoError(t, err)
	require.Len(t, results, 2) // primary run (parent) + 1 subagent

	// The lexicographically smallest run (run_alpha) is the parent and
	// carries its own messages so the experiment stays visible.
	parent := results[0].Session
	assert.Equal(t, "kernelowl:autoresearch_20260601_101801_def", parent.ID)
	assert.Empty(t, parent.ParentSessionID)
	assert.Equal(t, 1, parent.MessageCount)
	assert.Equal(t, "alpha q", parent.FirstMessage)

	child := results[1].Session
	assert.Equal(t,
		"kernelowl:autoresearch_20260601_101801_def/run_beta", child.ID)
	assert.Equal(t, parent.ID, child.ParentSessionID)
	assert.Equal(t, RelSubagent, child.RelationshipType)
	assert.Equal(t, "beta q", child.FirstMessage)
}

func TestParseKernelOwlExperimentEmptyDirReturnsNil(t *testing.T) {
	dir := t.TempDir()
	expDir := filepath.Join(dir, "empty_exp")
	require.NoError(t, os.MkdirAll(expDir, 0o755))
	kwWriteMeta(t, expDir, "empty_exp", "chat") // metadata but no transcripts

	results, err := ParseKernelOwlExperiment(expDir, "", "local")
	require.NoError(t, err)
	assert.Nil(t, results)
}

func TestDiscoverKernelOwlSessions(t *testing.T) {
	root := t.TempDir()
	// Repo layout: <root>/.kernelowl/experiments/<exp>
	expRoot := filepath.Join(root, ".kernelowl", "experiments")
	real := filepath.Join(expRoot, "autoresearch_20260601_101801_real")
	require.NoError(t, os.MkdirAll(real, 0o755))
	kwWriteConversation(t, real, "conversation_conv_x.json", "",
		[]kwMessage{{Role: "user", Content: "hi", Timestamp: "2026-06-01T10:00:00"}})

	// Underscore-prefixed (test traces) and hidden dirs are skipped
	// even when they contain transcripts.
	traces := filepath.Join(expRoot, "_traces", "test_thing")
	require.NoError(t, os.MkdirAll(traces, 0o755))
	kwWriteConversation(t, traces, "conversation_conv_t.json", "",
		[]kwMessage{{Role: "user", Content: "test", Timestamp: "2026-06-01T10:00:00"}})
	hidden := filepath.Join(expRoot, ".kernelowl", "nested")
	require.NoError(t, os.MkdirAll(hidden, 0o755))
	kwWriteConversation(t, hidden, "conversation_conv_h.json", "",
		[]kwMessage{{Role: "user", Content: "h", Timestamp: "2026-06-01T10:00:00"}})

	// A folder with no transcripts is skipped.
	require.NoError(t, os.MkdirAll(filepath.Join(expRoot, "no_transcripts"), 0o755))

	files := DiscoverKernelOwlSessions(root)
	require.Len(t, files, 1)
	assert.Equal(t, real, files[0].Path)
	assert.Equal(t, AgentKernelOwl, files[0].Agent)
}

func TestDiscoverKernelOwlSessionsAcceptsExperimentsRoot(t *testing.T) {
	// When pointed directly at the experiments dir (no descent),
	// immediate children are still discovered.
	expRoot := t.TempDir()
	exp := filepath.Join(expRoot, "chat_20260524_085006_x")
	require.NoError(t, os.MkdirAll(exp, 0o755))
	kwWriteConversation(t, exp, "conversation_conv_y.json", "",
		[]kwMessage{{Role: "user", Content: "hi", Timestamp: "2026-05-24T08:50:06"}})

	files := DiscoverKernelOwlSessions(expRoot)
	require.Len(t, files, 1)
	assert.Equal(t, exp, files[0].Path)
}

func TestFindKernelOwlSourceFile(t *testing.T) {
	root := t.TempDir()
	expRoot := filepath.Join(root, ".kernelowl", "experiments")
	exp := filepath.Join(expRoot, "autoresearch_20260601_101801_z")
	require.NoError(t, os.MkdirAll(exp, 0o755))

	// Repo-root input resolves via .kernelowl/experiments descent.
	assert.Equal(t, exp,
		FindKernelOwlSourceFile(root, "autoresearch_20260601_101801_z"))
	// Subagent id strips the /<runID> suffix before resolving.
	assert.Equal(t, exp,
		FindKernelOwlSourceFile(root, "autoresearch_20260601_101801_z/run_alpha"))
	// Experiments-root input resolves directly.
	assert.Equal(t, exp,
		FindKernelOwlSourceFile(expRoot, "autoresearch_20260601_101801_z"))
}

func TestKernelOwlExperimentInfoComposite(t *testing.T) {
	dir := t.TempDir()
	exp := filepath.Join(dir, "exp")
	require.NoError(t, os.MkdirAll(exp, 0o755))
	info, err := KernelOwlExperimentInfo(exp)
	require.NoError(t, err)

	// Adding a transcript changes the composite size and mtime.
	kwWriteConversation(t, exp, "conversation_conv_1.json", "",
		[]kwMessage{{Role: "user", Content: "first", Timestamp: "2026-06-01T10:00:00"}})
	info2, err := KernelOwlExperimentInfo(exp)
	require.NoError(t, err)
	assert.Greater(t, info2.Size(), info.Size(),
		"transcript must contribute to composite size")
}

// jsonResult parses a JSON string into a gjson.Result for tests.
func jsonResult(t *testing.T, raw string) gjson.Result {
	t.Helper()
	return gjson.Parse(raw)
}
