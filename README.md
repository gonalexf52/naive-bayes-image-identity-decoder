# Naive Bayes Image Identity Decoder

A Naive Bayes Decoder developed for OpenScope's Psycode Dataset—a project dedicated to investigating the effect of psilocybin administration on visual processing and behavior in the visual cortex of mice. 

# Pipeline

- Loads neural spike-train data in Neurodata Without Borders (NWB) format.
- Filters for quality-controlled single units within the selected visual cortex regions.
- Selects non-omitted natural image presentations from the pre-injection or post-injection periods, depending on user settings
- Counts spikes from 50-250 ms after stimulus onset, using 5 ms bins to compute the total spike count per neuron per trial.
- Partitions the data with a stratified random split, with 80% used for training and 20% reserved for testing
- Prints training accuracy, test accuracy, mean test confidence, and a confusion matrix

# Set Up

Install required dependencies:

- pip install numpy pandas pynwb scikit-learn

Download an NWB recording from the DANDI dataset and update these settings in naive-bayes-image-identity-decoder.py:

- SEGMENT = “pre” or “post”
- NWB_PATH = Path(r"path/to/your/recording.nwb")

Run:

- python naive-bayes-image-identity-decoder.py

# Citation

Amaya, A., Bawany, A., Buice, M., Carlson, M., De Filippo, R., Durand, S., Gillis, R., Grasso, C., Groblewski, P., Han, W., Johnson, T., Kenney, J., Kiselycznyk, C., Koch, C., Lecoq, J. A., Loeffler, H., Marks, L., Naidoo, R., Ott, T., Ouellette, B., Peene, R. C., Rembado, I., Schmitz, D., Suarez, L., Swapp, J., Weber, J., Wilkes, J., Williford, A., & Wyrick, D. (2026). Allen Institute OpenScope - Psycode (Version 0.260626.2154) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.001417/0.260626.2154

