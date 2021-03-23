# Metadata
Things are little bit messy, so I want to explain what all these files mean:
```bash
.../DSCI-550-Assignment-2/data/additional-features-v2 anthony-branch*
❯ tree -L 2
.
├── additional-features.json # this was the additional data json file in the firebase in assignment 1
├── additional_features.tsv # this was the additional data TSV file in assignment 1
├── new
│   ├── additional_features_TTR.tsv # this removes all the emails with no content in the TSV file, and do a Text-to-tag ratio algorithm to xhtml tika parsed
│   └── additional_features_w_content.tsv # this removes all the emails with no content in the TSV file
└── tika-similarity.tsv # this was the additional data + some tika parsed attrs in TSV in assignment 1

1 directory, 5 files
```

