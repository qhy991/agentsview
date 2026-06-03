package server

import (
	"net/http"

	"go.kenn.io/agentsview/internal/db"
)

func (s *Server) handleSessionUsage(
	w http.ResponseWriter, r *http.Request,
) {
	usage, err := s.db.GetSessionUsage(r.Context(), r.PathValue("id"))
	if err != nil {
		if handleContextError(w, err) {
			return
		}
		writeSessionUsageError(
			w,
			http.StatusInternalServerError,
			"usage_query_failed",
			"failed to query session usage",
		)
		return
	}
	if usage == nil {
		writeSessionUsageError(
			w,
			http.StatusNotFound,
			"session_not_found",
			"session not found",
		)
		return
	}

	writeJSON(w, http.StatusOK, newSessionUsageResponse(usage))
}

func newSessionUsageResponse(usage *db.SessionUsage) map[string]any {
	unpricedModels := usage.UnpricedModels
	if unpricedModels == nil {
		unpricedModels = []string{}
	}
	return map[string]any{
		"session_id":          usage.SessionID,
		"agent":               usage.Agent,
		"project":             usage.Project,
		"total_output_tokens": usage.TotalOutputTokens,
		"peak_context_tokens": usage.PeakContextTokens,
		"has_token_data":      usage.HasTokenData,
		"cost_usd":            usage.CostUSD,
		"has_cost":            usage.HasCost,
		"models":              usage.Models,
		"unpriced_models":     unpricedModels,
		"server_running":      true,
	}
}

func writeSessionUsageError(
	w http.ResponseWriter,
	status int,
	code string,
	message string,
) {
	writeJSON(w, status, map[string]any{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	})
}
