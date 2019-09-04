## ZaG2P
Convert non-Vietnamese word to Vietnamese phonemes/syllables

## Usage

### Install
`pip install https://github.com/enamoria/ZaG2P/zipball/master --verbose`

Download model from `https://drive.google.com/open?id=1liYQWBkN1uVAnguH6Bhsve4ptUQd_Myo`

### Example

    from ZaG2P.api import load_model, G2S  # Grapheme to syllables
    model, vietdict = load_model(fields_path, model_path, dict_path)  # fields_path and dict_path are optional, model_path is required

    start = time.time()
    G2S("hello", model, vietdict)
    print("Elapsed time: {}".format(time.time() - start))

    >> hello he lâu
    >> Elapsed time: 0.0081000328064

### Notes

* k, c ambiguity
* d, j ambiguity
