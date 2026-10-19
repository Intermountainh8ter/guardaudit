

## ShieldGemma prompt protocol (recorded 2026-10-19, second live audit)

Amendment 2 scores ShieldGemma 2B with all four standard principles in one prompt-only request and
reads P(Yes) once. Google's model card evaluates one policy per request. A model that answers "No"
when asked about four policies at once may answer "Yes" when asked about the one that applies, so
the recall reported here (0.31 and 0.44) is a lower bound on what a per-policy protocol would
find. The verdict stands as pre-registered; a per-policy variant (maximum P(Yes) over the four
single-policy prompts) would be a new numbered section applied only to audits committed after it.
Separately, the weights are a community GGUF conversion of a gated model; provenance is stated in
the result file and the original was not available on this machine.
