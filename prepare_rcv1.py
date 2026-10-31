
import gzip
import json
import random
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_DIR = Path("RCV1_original")
OUTPUT_DIR = Path("RCV1_CORE")

QRELS_FILE = INPUT_DIR / "rcv1-v2.topics.qrels.gz"

TRAIN_FILE = INPUT_DIR / "lyrl2004_tokens_train.dat.gz"

TEST_FILES = [
    INPUT_DIR / "lyrl2004_tokens_test_pt0.dat.gz",
    INPUT_DIR / "lyrl2004_tokens_test_pt1.dat.gz",
    INPUT_DIR / "lyrl2004_tokens_test_pt2.dat.gz",
    INPUT_DIR / "lyrl2004_tokens_test_pt3.dat.gz",
]

VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

EXPECTED_DOCUMENTS = 804414
EXPECTED_TOPICS = 103


# ============================================================
# 2. READ QRELS
# ============================================================

def read_qrels(path):
    """
    Read RCV1-V2 topic qrels.

    Supports common TREC-style rows:
        0 2286 C11 1

    Also supports three-column rows:
        C11 2286 1
        2286 C11 1
    """

    labels_by_doc = defaultdict(set)
    topic_codes = set()

    invalid_rows = []
    sample_rows = []

    def clean_token(token):
        return token.strip(" \t\r\n,;:\"'()[]{}")

    def is_topic_token(token):
        token = clean_token(token).upper()

        # Topic codes contain letters and may contain digits.
        return (
            bool(token)
            and any(c.isalpha() for c in token)
            and token.replace("-", "").isalnum()
        )

    with gzip.open(
        path, "rt",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for line_number, line in enumerate(f, 1):

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            if len(sample_rows) < 10:
                sample_rows.append(parts)

            # Last field is the relevance value.
            try:
                relevance = float(parts[-1])
            except ValueError:
                invalid_rows.append((line_number, parts))
                continue

            if relevance <= 0:
                continue

            # Find the topic code.
            topic_candidates = [
                clean_token(x).upper()
                for x in parts[:-1]
                if is_topic_token(x)
            ]

            # Find numeric fields excluding the final relevance.
            numeric_candidates = [
                clean_token(x)
                for x in parts[:-1]
                if clean_token(x).isdigit()
            ]

            if not topic_candidates or not numeric_candidates:
                invalid_rows.append((line_number, parts))
                continue

            # In four-column TREC format:
            # iteration, document_id, topic, relevance
            if len(parts) == 4 and parts[2].isalpha() is False:
                # Fall through to the general topic-token selection.
                pass

            topic = topic_candidates[-1]

            # In the standard four-column format, the document ID
            # is the second field. In three-column formats, it is
            # the numeric field before the topic or the first field.
            if len(parts) == 4 and is_topic_token(parts[2]):
                doc_id = clean_token(parts[1])
            elif is_topic_token(parts[0]):
                doc_id = clean_token(parts[1])
            elif is_topic_token(parts[1]):
                doc_id = clean_token(parts[0])
            else:
                invalid_rows.append((line_number, parts))
                continue

            labels_by_doc[doc_id].add(topic)
            topic_codes.add(topic)

    print("\n========== QRELS DIAGNOSTICS ==========")
    print("Sample rows:")
    for row in sample_rows:
        print(row)

    print("\nDocuments with positive labels:", len(labels_by_doc))
    print("Unique topic codes:", len(topic_codes))
    print("Topic codes:", sorted(topic_codes))

    print("\nUnrecognized rows:", len(invalid_rows))
    for row in invalid_rows[:5]:
        print(row)

    if len(topic_codes) != 103:
        raise ValueError(
            f"Expected 103 topics, but found {len(topic_codes)}. "
            "Check the printed sample rows and topic codes."
        )

    return labels_by_doc, sorted(topic_codes)



# ============================================================
# 3. READ ORIGINAL TOKEN FILES
# ============================================================

def read_token_file(path):
    """
    Read original LYRL2004 tokenized documents.

    Standard document structure:

        .I 2286
        .W
        token token token ...

    Also supports a simple alternative format:

        2286 token token token ...

    Returns:
        {document_id: document_text}
    """

    documents = {}

    current_id = None
    current_tokens = []
    reading_text = False

    def save_document():
        if current_id is not None:
            text = " ".join(current_tokens).strip()
            documents[current_id] = text

    print("Reading:", path.name)

    with gzip.open(path, "rt", encoding="utf-8",
                   errors="replace") as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            # New document marker.
            if line.startswith(".I"):

                save_document()

                parts = line.split(maxsplit=1)

                if len(parts) == 2:
                    current_id = parts[1].strip()
                else:
                    current_id = None

                current_tokens = []
                reading_text = False
                continue

            # Text section marker.
            if line.startswith(".W"):
                reading_text = True
                continue

            # Ignore other metadata sections.
            if line.startswith("."):
                reading_text = False
                continue

            # Standard .I / .W format.
            if current_id is not None:
                if reading_text:
                    current_tokens.append(line)
                continue

            # Alternative one-document-per-line format.
            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                doc_id, text = parts
                documents[doc_id] = text

    save_document()

    print("Documents loaded:", len(documents))

    return documents


# ============================================================
# 4. LOAD ALL DOCUMENTS
# ============================================================

def load_all_documents():

    print("\nLoading training documents...")

    train_documents = read_token_file(TRAIN_FILE)

    print("\nLoading test documents...")

    test_documents = {}

    for file_path in TEST_FILES:
        part = read_token_file(file_path)
        test_documents.update(part)

    print("\nDocument statistics")
    print("-------------------")
    print("Training documents:", len(train_documents))
    print("Test documents:", len(test_documents))
    print("Total documents:",
          len(train_documents) + len(test_documents))

    return train_documents, test_documents


# ============================================================
# 5. ATTACH LABELS TO DOCUMENTS
# ============================================================

def attach_labels(documents, labels_by_doc, topic_list):

    result = []

    for doc_id, text in documents.items():

        if not text.strip():
            continue

        # Keep only known topic labels.
        labels = sorted(
            labels_by_doc.get(doc_id, set())
        )

        # Convert labels into a fixed-size multi-hot vector.
        result.append({
            "id": doc_id,
            "text": text,
            "labels": labels
        })

    return result


# ============================================================
# 6. SPLIT TRAINING DATA INTO TRAIN AND VALIDATION
# ============================================================

def split_train_validation(data):

    data = list(data)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(data)

    valid_size = int(len(data) * VALIDATION_RATIO)

    valid_data = data[:valid_size]
    train_data = data[valid_size:]

    return train_data, valid_data


# ============================================================
# 7. SAVE DATA IN CORE FORMAT
# ============================================================

def save_split(data, split_name, topic_to_id):

    src_path = OUTPUT_DIR / f"{split_name}.src"
    tgt_path = OUTPUT_DIR / f"{split_name}.tgt"

    with open(src_path, "w", encoding="utf-8") as src_file, \
         open(tgt_path, "w", encoding="utf-8") as tgt_file:

        for item in data:

            # Original pre-tokenized document text.
            src_file.write(
                item["text"].replace("\n", " ") + "\n"
            )

            # Topic codes, separated by spaces.
            tgt_file.write(
                " ".join(item["labels"]) + "\n"
            )

    print(f"{split_name}: {len(data)} documents saved")


# ============================================================
# 8. SAVE LABEL VOCABULARY
# ============================================================

def save_label_vocabulary(topic_list):

    topic_to_id = {
        topic: index
        for index, topic in enumerate(topic_list)
    }

    with open(
        OUTPUT_DIR / "topic_sorted.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(topic_list, f, indent=2)

    with open(
        OUTPUT_DIR / "label_to_id.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(topic_to_id, f, indent=2)

    return topic_to_id


# ============================================================
# 9. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Read qrels.
    labels_by_doc, topic_list = read_qrels(QRELS_FILE)

    # Step 2: Load tokenized documents.
    train_documents, test_documents = load_all_documents()

    # Step 3: Attach topic labels.
    train_data = attach_labels(
        train_documents,
        labels_by_doc,
        topic_list
    )

    test_data = attach_labels(
        test_documents,
        labels_by_doc,
        topic_list
    )

    # Step 4: Create validation split from training documents.
    train_data, valid_data = split_train_validation(train_data)

    # Step 5: Save label vocabulary.
    topic_to_id = save_label_vocabulary(topic_list)

    # Step 6: Save datasets.
    save_split(train_data, "train", topic_to_id)
    save_split(valid_data, "valid", topic_to_id)
    save_split(test_data, "test", topic_to_id)

    # Step 7: Summary.
    print("\n================================")
    print("RCV1-V2 preprocessing completed")
    print("================================")

    print("Number of labels:", len(topic_list))
    print("Training:", len(train_data))
    print("Validation:", len(valid_data))
    print("Testing:", len(test_data))

    print("\nOutput directory:", OUTPUT_DIR.resolve())

    print("\nExpected output files:")
    print("train.src")
    print("train.tgt")
    print("valid.src")
    print("valid.tgt")
    print("test.src")
    print("test.tgt")
    print("topic_sorted.json")
    print("label_to_id.json")


if __name__ == "__main__":
    main()
