// ABOUTME: Parses KernelOwl experiment directories into sessions.
// ABOUTME: Each experiment folder is one parent session; multi-run
// experiments additionally yield one subagent session per run.
package parser

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

// KernelOwl stores sessions as folders, not files: each experiment
// directory under .kernelowl/experiments/ holds a parent session and,
// for multi-agent experiments, one subfolder per run. Transcripts are
// OpenAI-style JSON (conversation_conv_*.json) carrying a metadata
// block that self-describes the owning experiment (session_dir) and
// run (task_id), so a folder walk can route every transcript to the
// right session.

// kernelOwlMeta is the subset of experiment-level metadata the
// parser uses to build the parent session identity.
type kernelOwlMeta struct {
	id          string
	kind        string // session_kind: autoresearch, chat, assistant, ...
	name        string // assistant_name or description
	description string
	status      string
	started     time.Time
	ended       time.Time
}

// ParseKernelOwlExperiment parses a KernelOwl experiment directory
// and returns one ParseResult for the experiment (parent) plus one
// per run when the experiment has multiple sub-agents. Directories
// with no transcripts return nil so discovery/sync skip them.
func ParseKernelOwlExperiment(
	dir, project, machine string,
) ([]ParseResult, error) {
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", dir)
	}

	name := filepath.Base(dir)
	meta := loadKernelOwlMeta(dir, name)
	transcripts := findKernelOwlTranscripts(dir)
	if len(transcripts) == 0 {
		return nil, nil
	}

	// Route each transcript to the experiment (no task_id) or to the
	// run identified by its metadata task_id.
	var parentMsgs []ParsedMessage
	runs := make(map[string][]ParsedMessage)
	for _, path := range transcripts {
		msgs, taskID := parseKernelOwlTranscript(path)
		if len(msgs) == 0 {
			continue
		}
		if taskID != "" {
			runs[taskID] = append(runs[taskID], msgs...)
		} else {
			parentMsgs = append(parentMsgs, msgs...)
		}
	}

	// Sort run ids deterministically. The lexicographically smallest
	// id (run_001_main_...) is the natural primary run.
	runIDs := make([]string, 0, len(runs))
	for id := range runs {
		runIDs = append(runIDs, id)
	}
	sort.Strings(runIDs)

	if len(runIDs) <= 1 {
		// Zero or one run: fold every transcript into a single parent
		// session so the experiment is one visible session.
		for _, id := range runIDs {
			parentMsgs = append(parentMsgs, runs[id]...)
		}
		parent := buildKernelOwlSession(
			dir, project, machine, meta, parentMsgs, "", "",
		)
		return []ParseResult{{Session: parent, Messages: parentMsgs}}, nil
	}

	// Multiple runs: the primary run (first, typically run_001_main)
	// becomes the parent session carrying its own messages so the
	// experiment shows up in the session list, which hides both empty
	// sessions and bare subagents. The remaining runs are subagents
	// linked to it. Each run's messages live in exactly one session,
	// so token totals do not double-count.
	primary := runIDs[0]
	parentMsgs = append(parentMsgs, runs[primary]...)
	parent := buildKernelOwlSession(
		dir, project, machine, meta, parentMsgs, "", "",
	)
	results := []ParseResult{{Session: parent, Messages: parentMsgs}}
	for _, runID := range runIDs[1:] {
		msgs := runs[runID]
		child := buildKernelOwlSession(
			dir, project, machine, meta,
			msgs, parent.ID, runID,
		)
		results = append(results, ParseResult{
			Session: child, Messages: msgs,
		})
	}
	return results, nil
}

// buildKernelOwlSession assembles a ParsedSession for either the
// parent experiment or a run subagent. parentID is empty for the
// parent; runID is empty for the parent.
func buildKernelOwlSession(
	dir, project, machine string,
	meta kernelOwlMeta,
	msgs []ParsedMessage,
	parentID, runID string,
) ParsedSession {
	sort.SliceStable(msgs, func(i, j int) bool {
		return msgs[i].Timestamp.Before(msgs[j].Timestamp)
	})
	for i := range msgs {
		msgs[i].Ordinal = i
	}

	started, ended := meta.started, meta.ended
	var firstMessage string
	var userCount int
	for _, m := range msgs {
		if started.IsZero() || m.Timestamp.Before(started) {
			started = m.Timestamp
		}
		if m.Timestamp.After(ended) {
			ended = m.Timestamp
		}
		if m.Role == RoleUser && !m.IsSystem && m.Content != "" {
			userCount++
			if firstMessage == "" {
				firstMessage = truncate(
					strings.ReplaceAll(m.Content, "\n", " "), 300,
				)
			}
		}
	}

	id := "kernelowl:" + meta.id
	if runID != "" {
		id = "kernelowl:" + meta.id + "/" + runID
	}

	sess := ParsedSession{
		ID:               id,
		Project:          deriveKernelOwlProject(project, meta),
		Machine:          machine,
		Agent:            AgentKernelOwl,
		SessionName:      meta.name,
		FirstMessage:     firstMessage,
		StartedAt:        started,
		EndedAt:          ended,
		MessageCount:     len(msgs),
		UserMessageCount: userCount,
		File: FileInfo{
			Path: dir,
		},
	}
	if parentID != "" {
		sess.ParentSessionID = parentID
		sess.RelationshipType = RelSubagent
	}
	accumulateMessageTokenUsage(&sess, msgs)
	return sess
}

// deriveKernelOwlProject picks a Project label: an explicit passed-in
// value wins, then the experiment kind, then the folder-name prefix.
func deriveKernelOwlProject(project string, meta kernelOwlMeta) string {
	if project != "" {
		return project
	}
	if meta.kind != "" {
		return meta.kind
	}
	if i := strings.IndexByte(meta.id, '_'); i > 0 {
		return meta.id[:i]
	}
	return "kernelowl"
}

// loadKernelOwlMeta reads experiment metadata from the first
// available metadata file across the layouts KernelOwl uses
// (session_metadata.json at the root or under session_core/, or
// session_core/session_state_v2.json). Falls back to folder name.
func loadKernelOwlMeta(dir, name string) kernelOwlMeta {
	meta := kernelOwlMeta{id: name}
	for _, candidate := range []string{
		filepath.Join(dir, "session_metadata.json"),
		filepath.Join(dir, "session_core", "session_metadata.json"),
	} {
		if data, err := os.ReadFile(candidate); err == nil {
			fillKernelOwlMeta(&meta, data)
			if meta.kind != "" || meta.started != (time.Time{}) {
				return meta
			}
		}
	}
	// session_state_v2.json carries less identity but may be the
	// only artifact present (autoresearch layout).
	if data, err := os.ReadFile(
		filepath.Join(dir, "session_core", "session_state_v2.json"),
	); err == nil {
		if t := parseKernelOwlTime(
			gjson.GetBytes(data, "updated_at").Str,
		); !t.IsZero() {
			meta.ended = t
		}
	}
	return meta
}

func fillKernelOwlMeta(meta *kernelOwlMeta, data []byte) {
	root := gjson.ParseBytes(data)
	if v := root.Get("session_id").Str; v != "" {
		meta.id = v
	}
	meta.kind = root.Get("session_kind").Str
	if meta.kind == "" {
		meta.kind = root.Get("kind").Str
	}
	meta.name = root.Get("assistant_name").Str
	if meta.name == "" {
		meta.name = root.Get("name").Str
	}
	meta.description = root.Get("description").Str
	meta.status = root.Get("status").Str
	meta.started = parseKernelOwlTime(root.Get("created_at").Str)
	if t := parseKernelOwlTime(root.Get("updated_at").Str); !t.IsZero() {
		meta.ended = t
	}
}

// findKernelOwlTranscripts returns the canonical
// conversation_conv_*.json transcripts under dir. llm_call_*.json
// single-call logs and rendered kernelowl_*.json conversations are
// intentionally excluded to avoid double-counting.
func findKernelOwlTranscripts(dir string) []string {
	var paths []string
	_ = filepath.WalkDir(dir, func(
		path string, d os.DirEntry, err error,
	) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		base := d.Name()
		if !strings.HasPrefix(base, "conversation_conv_") ||
			!strings.HasSuffix(base, ".json") {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	sort.Strings(paths)
	return paths
}

// parseKernelOwlTranscript parses one conversation_conv_*.json file
// into messages and returns the run task_id from its metadata block
// (empty for experiment-level conversations).
func parseKernelOwlTranscript(
	path string,
) (msgs []ParsedMessage, taskID string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, ""
	}
	if !gjson.ValidBytes(data) {
		return nil, ""
	}
	root := gjson.ParseBytes(data)
	taskID = root.Get("metadata.task_id").Str

	messages := root.Get("messages")
	if !messages.IsArray() {
		return nil, taskID
	}
	msgs = make([]ParsedMessage, 0)
	messages.ForEach(func(_, msg gjson.Result) bool {
		parsed, ok := parseKernelOwlMessage(msg)
		if !ok {
			return true
		}
		msgs = append(msgs, parsed)
		return true
	})
	return msgs, taskID
}

// parseKernelOwlMessage maps a KernelOwl transcript message (OpenAI
// chat shape with a usage block) onto ParsedMessage. System messages
// are flagged IsSystem so they are excluded from user-turn counts and
// search but still rendered in the transcript.
func parseKernelOwlMessage(msg gjson.Result) (ParsedMessage, bool) {
	role := msg.Get("role").Str
	content := msg.Get("content").Str
	if strings.TrimSpace(content) == "" && role != "system" {
		return ParsedMessage{}, false
	}

	var parsedRole RoleType
	var isSystem bool
	switch role {
	case "assistant":
		parsedRole = RoleAssistant
	case "system":
		// No RoleSystem exists; system turns ride on the user role
		// flagged IsSystem, matching how the Claude/ChatGPT parsers
		// represent injected system content.
		parsedRole = RoleUser
		isSystem = true
	default:
		parsedRole = RoleUser
	}

	usage := msg.Get("usage")
	var tokenUsage []byte
	if usage.Exists() {
		tokenUsage = []byte(usage.Raw)
	}
	contextTokens := int(usage.Get("input_tokens").Int()) +
		int(usage.Get("cache_creation_input_tokens").Int()) +
		int(usage.Get("cache_read_input_tokens").Int())
	outputTokens := int(usage.Get("output_tokens").Int())

	return ParsedMessage{
		Role:          parsedRole,
		IsSystem:      isSystem,
		Content:       content,
		ContentLength: len(content),
		Timestamp:     parseKernelOwlTime(msg.Get("timestamp").Str),
		Model:         msg.Get("model").Str,
		TokenUsage:    tokenUsage,
		ContextTokens: contextTokens,
		OutputTokens:  outputTokens,
		HasContextTokens: usage.Get("input_tokens").Exists() ||
			usage.Get("cache_creation_input_tokens").Exists() ||
			usage.Get("cache_read_input_tokens").Exists(),
		HasOutputTokens:    usage.Get("output_tokens").Exists(),
		tokenPresenceKnown: true,
	}, true
}

// parseKernelOwlTime parses KernelOwl timestamps, which may be
// RFC3339 ("2026-05-23T10:24:50Z") or naive datetimes with fractional
// seconds and no timezone ("2026-03-19T10:35:17.980467"). Naive
// values are interpreted as UTC.
func parseKernelOwlTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	for _, layout := range []string{
		time.RFC3339Nano,
		time.RFC3339,
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC()
		}
	}
	for _, layout := range []string{
		"2006-01-02T15:04:05.999999",
		"2006-01-02T15:04:05.999",
		"2006-01-02T15:04:05",
	} {
		if t, err := time.ParseInLocation(
			layout, s, time.UTC,
		); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

// KernelOwlExperimentInfo returns a composite os.FileInfo for an
// experiment directory: the size is the sum and the mtime the max
// across the experiment's metadata and transcript files. A single
// anchor stat would miss changes confined to transcripts or runs, so
// change detection needs this composite (same approach as
// AntigravityFileInfo for .db + sidecars).
func KernelOwlExperimentInfo(dir string) (os.FileInfo, error) {
	base, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	var size int64
	var mtime int64
	consider := func(path string) {
		info, err := os.Stat(path)
		if err != nil {
			return
		}
		size += info.Size()
		if t := info.ModTime().UnixNano(); t > mtime {
			mtime = t
		}
	}
	for _, rel := range []string{
		"session_metadata.json",
		"session_core/session_metadata.json",
		"session_core/session_state_v2.json",
	} {
		consider(filepath.Join(dir, rel))
	}
	for _, t := range findKernelOwlTranscripts(dir) {
		consider(t)
	}
	if size == 0 {
		// Empty or unreadable: fall back to the directory stat so the
		// caller still gets a usable FileInfo.
		return base, nil
	}
	return kernelOwlFileInfo{
		name:  base.Name(),
		size:  size,
		mtime: mtime,
	}, nil
}

// DiscoverKernelOwlSessions finds KernelOwl experiment directories.
// The configured directory may be either the experiments root
// (".kernelowl/experiments") or the KernelOwl repo root; in the
// latter case discovery descends into ".kernelowl/experiments" when
// present. Each immediate subdirectory that holds at least one
// transcript is emitted as one session anchored on the directory.
// Underscore-prefixed ("_traces") and hidden (".kernelowl") folders
// are skipped.
func DiscoverKernelOwlSessions(dir string) []DiscoveredFile {
	if dir == "" {
		return nil
	}
	root := dir
	if experiments := filepath.Join(
		dir, ".kernelowl", "experiments",
	); isDir(experiments) {
		root = experiments
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var files []DiscoveredFile
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") {
			continue
		}
		expDir := filepath.Join(root, name)
		if !hasKernelOwlTranscript(expDir) {
			continue
		}
		files = append(files, DiscoveredFile{
			Path:  expDir,
			Agent: AgentKernelOwl,
		})
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].Path < files[j].Path
	})
	return files
}

// FindKernelOwlSourceFile resolves a KernelOwl session ID (prefix
// already stripped) to its experiment directory. Subagent IDs carry
// a "/<runID>" suffix; only the experiment segment maps to a path.
func FindKernelOwlSourceFile(dir, sessionID string) string {
	if dir == "" || sessionID == "" {
		return ""
	}
	expName := sessionID
	if i := strings.IndexByte(expName, '/'); i >= 0 {
		expName = expName[:i]
	}
	expDir := filepath.Join(dir, expName)
	if isDir(expDir) {
		return expDir
	}
	// sessionID may have been resolved against a repo root rather
	// than the experiments root.
	if experiments := filepath.Join(
		dir, ".kernelowl", "experiments", expName,
	); isDir(experiments) {
		return experiments
	}
	return ""
}

// hasKernelOwlTranscript reports whether dir contains any
// conversation_conv_*.json transcript, short-circuiting on the first
// match so discovery does not walk entire experiment trees.
func hasKernelOwlTranscript(dir string) bool {
	found := false
	_ = filepath.WalkDir(dir, func(
		path string, d os.DirEntry, err error,
	) error {
		if err != nil || d.IsDir() {
			return nil
		}
		base := d.Name()
		if strings.HasPrefix(base, "conversation_conv_") &&
			strings.HasSuffix(base, ".json") {
			found = true
			return filepath.SkipDir
		}
		return nil
	})
	return found
}

func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// kernelOwlFileInfo is a minimal os.FileInfo for composite stats.
type kernelOwlFileInfo struct {
	name  string
	size  int64
	mtime int64
}

func (f kernelOwlFileInfo) Name() string       { return f.name }
func (f kernelOwlFileInfo) Size() int64        { return f.size }
func (f kernelOwlFileInfo) Mode() os.FileMode  { return 0o755 | os.ModeDir }
func (f kernelOwlFileInfo) ModTime() time.Time { return time.Unix(0, f.mtime).UTC() }
func (f kernelOwlFileInfo) IsDir() bool        { return true }
func (f kernelOwlFileInfo) Sys() any           { return nil }
