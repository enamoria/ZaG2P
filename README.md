## ZaG2P
Convert non-Vietnamese word to Vietnamese phonemes/syllables

## Requirements
* `python 2.7`
* `torch == 1.1.0`
* `torchtext == 0.3.1`

## Usage

### Install
`pip install https://github.com/enamoria/ZaG2P/zipball/py3 --verbose`

### Example

The `Elapsed time` varies between runs, still don't know why

    from ZaG2P.api import load_model, G2S  # Grapheme to syllables
    model, vietdict = load_model()

    start = time.time()
    G2S("hello", model, vietdict)
    print("Elapsed time: {}".format(time.time() - start))

    >> hello he lâu
    >> Elapsed time: 0.0081000328064

## Notes

* k, c ambiguity
* d, j ambiguity

## References
Read this wonderful blog `https://fehiepsi.github.io/blog/grapheme-to-phoneme/`.
