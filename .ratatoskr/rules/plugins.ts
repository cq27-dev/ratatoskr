// Which plugins each node gets.
//
// A node with no ruleset — or one that says nothing about plugins — gets every plugin the config
// discovered. That is right for `rag-rat`, whose tools every node benefits from, and wrong for a
// plugin that changes how a node *thinks*: it would reach the characterizer transcribing test names
// as readily as the node writing the code.
//
// `ponytail` is a behavioural mode rather than a toolset. It arrives through its `SessionStart`
// hook, whose output is prefixed to the node's preamble, so it is on for the whole of that node's
// work instead of being a tool the node has to remember to ask for.

// The repository-wide set. Naming it is what stops `ponytail` reaching every node by default.
defineDefaults({ plugins: ["rag-rat"] });

// The two nodes that decide what to build and then build it. `add` rather than a bare list: a list
// replaces the defaults, so `rag-rat` would have to be repeated here and would fall out of these
// two nodes the day the default set changes.
defineAgent("analyst", {
  plugins: { add: ["ponytail"] },
  // `allow` replaces the stage defaults, so keep its repo and file-reading tools alongside search.
  tools: {
    allow: [
      "impact_surface",
      "symbol_lookup",
      "semantic_search",
      "Read",
      "Grep",
      "Glob",
      "WebSearch",
    ],
  },
});
defineAgent("implementer", {
  plugins: { add: ["ponytail"] },
  tools: {
    allow: [
      "impact_surface",
      "symbol_lookup",
      "semantic_search",
      "find_callers",
      "memory_search",
      "read_chunk",
      "memory_update",
      "memory_mark_obsolete",
      "Read",
      "Grep",
      "Glob",
      "Write",
      "Edit",
      "Bash",
      "ask",
      "WebSearch",
    ],
  },
});
