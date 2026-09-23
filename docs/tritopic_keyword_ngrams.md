# TriTopic keyword n-grams

Active TriTopic and FastTriTopic experiments explicitly use single-word keywords:

```yaml
model:
  params:
    keyword_ngram_range: [1, 1]
```

The bounds are inclusive: `[1, 2]` allows words and two-word phrases. The model
factories validate the pair and apply it to the keyword extractor before fitting.
This is a project-level option, removed before constructing the upstream
`TriTopicConfig`, which does not expose this setting. FastTriTopic inherits the
same extractor. Graph text features are unaffected.

The optimizer treats a flat pair as one setting; nested pairs such as
`[[1, 1], [1, 2]]` define a search grid. Configurations omitting the option retain
the upstream `(1, 2)` default. Archived experiments and existing results are not
rewritten. New results record the option in their resolved configuration.

Unigram extraction aligns keyword length with the BERTopic experiment defaults.
It does not by itself align coherence tokenization or guarantee finite scores.
