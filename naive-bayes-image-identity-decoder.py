from pathlib import Path
import numpy as np
from pynwb import NWBHDF5IO
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ----------------------------- User settings -----------------------------

SEGMENT = "pre"  

NWB_PATH = Path()

TIME_RESOLUTION = 0.005
WINDOW_START_TIME = 0.050
WINDOW_END_TIME = 0.250

BRAIN_REGIONS = [
    "VISp1",
    "VISp2/3",
    "VISp4",
    "VISp5",
    "VISp6a",
    "VISp6b",
    "VISam2/3",
    "VISam4",
    "VISam5",
    "VISam6a",
    "VISam6b",
]


# ---------------------------- Data functions -----------------------------

def get_unit_location(unit_row):
    """Assign a unit to the location of its peak waveform channel."""
    waveform_mins = np.min(unit_row["waveform_mean"], axis=0)
    peak_channel_index = np.argmin(waveform_mins)
    return unit_row["electrodes"].iloc[peak_channel_index].location


def select_units(units):
    """Apply the Psycode quality criteria and visual-area selection."""
    quality_units = units.query(
        '(firing_range > 8) & '
        '(decoder_label == "sua") & '
        '(presence_ratio > .95)'
    )

    selected_spike_times = []
    selected_locations = []

    for _, unit_row in quality_units.iterrows():
        location = get_unit_location(unit_row)
        if location in BRAIN_REGIONS:
            selected_spike_times.append(unit_row.spike_times)
            selected_locations.append(location)

    return selected_spike_times, selected_locations


def get_spike_matrix(stim_times, units_spike_times, bin_edges):
    """Return a units x trials x time-bins matrix of spike counts."""
    time_resolution = np.mean(np.diff(bin_edges))
    spike_matrix = np.zeros(
        (len(units_spike_times), len(stim_times), len(bin_edges) - 1),
        dtype=np.uint8,
    )

    for unit_index, spike_times in enumerate(units_spike_times):
        spike_times = np.asarray(spike_times)

        for trial_index, stim_time in enumerate(stim_times):
            first_time = stim_time + bin_edges[0]
            last_time = stim_time + bin_edges[-1]
            first_spike, last_spike = np.searchsorted(
                spike_times, [first_time, last_time]
            )
            spikes_in_window = spike_times[first_spike:last_spike]
            bin_indices = (
                (spikes_in_window - first_time) / time_resolution
            ).astype(int)

            for bin_index in bin_indices:
                if 0 <= bin_index < spike_matrix.shape[2]:
                    spike_matrix[unit_index, trial_index, bin_index] += 1

    return spike_matrix


# ----------------------------- Baseline model ----------------------------

class PoissonNB(ClassifierMixin, BaseEstimator):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if np.any(X < 0):
            raise ValueError("PoissonNB requires nonnegative spike counts.")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        class_counts = np.bincount(encoded_y)
        self.class_log_prior_ = np.log(class_counts / class_counts.sum())
        self.feature_rate_ = np.asarray(
            [
                X[encoded_y == class_index].mean(axis=0) + self.alpha
                for class_index in range(len(self.classes_))
            ]
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = (
            X @ np.log(self.feature_rate_).T
            - self.feature_rate_.sum(axis=1)
            + self.class_log_prior_
        )

        # Stable softmax: convert class log scores into probabilities.
        scores = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


# -------------------------- Load and organize data ------------------------

if SEGMENT not in {"pre", "post"}:
    raise ValueError('SEGMENT must be either "pre" or "post".')
if not NWB_PATH.is_file():
    raise FileNotFoundError(f"NWB file not found:\n{NWB_PATH}")

io = NWBHDF5IO(str(NWB_PATH), mode="r", load_namespaces=True)

try:
    nwb = io.read()
    units = nwb.units.to_dataframe()
    units_spike_times, unit_locations = select_units(units)

    stim_table = nwb.intervals["behavior_presentations"]
    all_stim_times = np.asarray(stim_table["start_time"][:], dtype=float)
    all_image_names = np.asarray(stim_table["image_name"][:]).astype(str)

    omitted_values = np.asarray(stim_table["omitted"][:])
    omitted = np.asarray(
        [str(value).strip().lower() == "true" for value in omitted_values],
        dtype=bool,
    )

    injection_time = (
        nwb.intervals["injection_times"]
        .to_dataframe()["start_time"]
        .iloc[0]
    )

    valid_images = (
        ~omitted
        & (all_image_names != "")
        & (all_image_names != "nan")
        & (all_image_names != "None")
    )

    repeated_image = np.zeros(len(all_image_names), dtype=bool)
    valid_indices = np.flatnonzero(valid_images)
    if len(valid_indices) > 1:
        current_indices = valid_indices[1:]
        previous_indices = valid_indices[:-1]
        repeated_image[current_indices] = (
            all_image_names[current_indices] == all_image_names[previous_indices]
        )

    if SEGMENT == "pre":
        segment_mask = all_stim_times < injection_time
    else:
        segment_mask = all_stim_times >= injection_time

    selected_trials = valid_images & segment_mask
    stim_times = all_stim_times[selected_trials]
    image_names = all_image_names[selected_trials]

    n_bins = int(
        (WINDOW_END_TIME - WINDOW_START_TIME) / TIME_RESOLUTION
    )
    bin_edges = np.linspace(
        WINDOW_START_TIME,
        WINDOW_END_TIME,
        n_bins + 1,
        endpoint=True,
    )
    spike_matrix = get_spike_matrix(
        stim_times, units_spike_times, bin_edges
    )

finally:
    io.close()

if len(units_spike_times) == 0:
    raise ValueError("No units passed the Psycode selection criteria.")
if len(stim_times) == 0:
    raise ValueError(f"No repeated images were found in the {SEGMENT} segment.")

print(f"Selected {len(units_spike_times)} units")
locations, location_counts = np.unique(unit_locations, return_counts=True)
for location, count in zip(locations, location_counts):
    print(f"{location}: {count}")
print("Spike matrix shape:", spike_matrix.shape)

X_all = spike_matrix.sum(axis=2).T
y_all = image_names

print(f"\n{SEGMENT.capitalize()}-injection repeated-image counts:")
image_classes, image_counts = np.unique(y_all, return_counts=True)
for image_name, count in zip(image_classes, image_counts):
    print(f"{image_name}: {count} trials")


# -------------------------- Train and evaluate ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=314159,
    stratify=y_all,
)

decoder = Pipeline(
    [
        ("remove_constant_units", VarianceThreshold(threshold=0)),
        ("poisson_nb", PoissonNB(alpha=0.01)),
    ]
)

decoder.fit(X_train, y_train)
y_train_predicted = decoder.predict(X_train)
y_test_predicted = decoder.predict(X_test)
test_probabilities = decoder.predict_proba(X_test)
test_confidence = test_probabilities.max(axis=1)
mean_test_confidence = test_confidence.mean()
test_accuracy = accuracy_score(y_test, y_test_predicted)
labels = decoder.named_steps["poisson_nb"].classes_
confusion = confusion_matrix(y_test, y_test_predicted, labels=labels)

print(f"\n{SEGMENT.capitalize()}-injection baseline image decoder")
print("-" * 48)
print(f"Trials: {len(y_all)}")
print(f"Training trials: {len(y_train)}")
print(f"Testing trials: {len(y_test)}")
print(f"Selected units: {len(units_spike_times)}")
print(f"Number of image classes: {len(labels)}")
print(f"Uniform random chance: {1 / len(labels):.4f}")
print(f"Training accuracy: {accuracy_score(y_train, y_train_predicted):.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(
    "Test balanced accuracy: "
    f"{balanced_accuracy_score(y_test, y_test_predicted):.4f}"
)
print(f"Mean test confidence: {mean_test_confidence:.4f}")
print(
    "Confidence minus accuracy: "
    f"{mean_test_confidence - test_accuracy:.4f}"
)
print("\nImage label order:")
print(labels)
print("\nConfusion matrix:")
print(confusion)
