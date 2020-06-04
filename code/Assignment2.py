# dataset function return [[word~tag, ...] [... ], ... ]
def dataset(path):
    sentence_list = []
    tag_and_word_sentence = []
    with open(path) as f:
        for line in f:
            if line != '-DOCSTART- -X- -X- O\n':
                if line != "\n":
                    token = line.split()
                    word = token[0]
                    tag = token[3]
                    tag_and_word_sentence.append(word + "~" + tag)
                else:
                    if len(tag_and_word_sentence) > 0:
                        sentence_list.append(tag_and_word_sentence)
                        tag_and_word_sentence = []
    return sentence_list


# Task 1: Build a Bi-gram Hidden Markov Model (HMM )
# In all three, I hold both counts and probabilities in different dictionary.
def hidden_markov_model(train_sentences):
    tags_and_count = {}  # all tag and frequency
    in_count = {}  # initial count
    in_prob = {}  # initial probability

    tr_count = {}  # transitional count
    tr_prob = {}  # transitional probability

    em_count = {}  # Emission  count
    em_prob = {}  # Emission  probability

    unique_word = {}  # all unique word in train dataset ( for smoothing )
    total_sent = len(train_sentences)  # all sentence count in train dataset (for initial probability )

    for sent in train_sentences:
        for i in range(len(sent)): # For Example sent[i] = Meltem~Person
            word = sent[i].split("~")[0].lower()
            tag = sent[i].split("~")[1]

            # unique word count--------------------------
            if word not in unique_word:
                unique_word[word] = 1
            else:
                unique_word[word] += 1

            # all tag and count--------------------------
            if tag not in tags_and_count:
                tags_and_count[tag] = 1
            else:
                tags_and_count[tag] += 1

            # initial count-----------------------------
            # like this {'B-ORG': 2455, ... }
            if i == 0:
                if tag in in_count:
                    in_count[tag] += 1
                else:
                    in_count[tag] = 1

            # transition count--------------------------
            # like this {'B-ORG': {'O': 3736, ...}, ... }
            if i < len(sent) - 1:
                pair = sent[i + 1].split("~")
                next_tag = pair[1]

            if tag not in tr_count.keys():
                tr_count[tag] = {next_tag: 1}
            else:
                if next_tag in tr_count[tag].keys():
                    tr_count[tag][next_tag] += 1
                else:
                    tr_count[tag][next_tag] = 1

            # emission count----------------------------
            # like this {'B-ORG': {'eu':24, 'european':29 ... }, ...}
            if tag not in em_count:
                em_count[tag] = {word: 1}
            else:
                if word in em_count[tag]:
                    em_count[tag][word] += 1
                else:
                    em_count[tag][word] = 1

    # total unique word = 21009 (train)
    total_unique_word = len(unique_word.keys())

    # initial probability and smoothing --------------
    for t in tags_and_count.keys():
        if t not in in_count:
            in_count[t] = 0
            in_prob[t] = 1 / (total_sent + 9)
        if in_count[t] != 0:
            in_prob[t] = in_count[t] / total_sent

    # transition probability and smoothing------------
    for t in tags_and_count.keys():
        for k in tr_count.keys():
            if t not in tr_count[k].keys():
                tr_count[k][t] = 0

    for k in tr_count.keys():
        tr_prob[k] = {}
        total_tag = 0
        for item in tr_count[k].keys():
            total_tag += tr_count[k][item]
        for item2 in tr_count[k].keys():
            if tr_count[k][item2] != 0:
                tr_prob[k][item2] = tr_count[k][item2] / total_tag
            else:
                tr_prob[k][item2] = 1 / (total_tag + 9)

    # emission probability------------------------------
    for k in em_count.keys():
        em_prob[k] = {}
        total_words = 0
        for item in em_count[k].keys():
            total_words += em_count[k][item]
        for item2 in em_count[k].keys():
            if em_count[k][item2] != 0:
                em_prob[k][item2] = em_count[k][item2] / total_words
            else:
                em_prob[k][item2] = 0

    return in_prob, tr_prob, em_prob, em_count, total_unique_word


# Each matrix element hava some attribute
class MatrixItem:
    def __init__(self, r_tag, c_word, probability, back_pointer):
        self.r_tag = r_tag
        self.c_word = c_word
        self.probability = probability
        self.back_pointer = back_pointer


def smoothing(em_count_dict, unique_word, tag):
    total_words_count = 0
    for item in em_count_dict[tag].keys():
        total_words_count += em_count_dict[tag][item]
    smooth_prob = 1 / (total_words_count + unique_word)
    return smooth_prob


# c_r = present row
# c_c = present_column
# Task 2: Viterbi Algorithm
def viterbi(c_r, c_c, matrix, initial_prob, transition_prob, emission_prob, emission_count, unique_word_count):
    back_pointer = -1
    max_probability = -1.0
    tag = matrix[c_r][c_c].r_tag
    word = matrix[c_r][c_c].c_word

    if c_c == 0:  # first word
        if word in emission_prob[tag] and emission_prob[tag][word] != 0:
            max_probability = (initial_prob[tag] * emission_prob[tag][word])
        else:  # smoothing doing
            max_probability = (initial_prob[tag] * smoothing(emission_count, unique_word_count, tag))
        back_pointer = -1
    else: # after first words
        for temp_row in range(9):  # each tag check
            change_tag = matrix[temp_row][c_c - 1].r_tag
            before_word = matrix[temp_row][c_c - 1].c_word
            if before_word in emission_prob[change_tag] and emission_prob[change_tag][before_word] != 0:
                prob = (emission_prob[change_tag][before_word] * transition_prob[tag][change_tag])
            else:  # smoothing doing
                prob = (smoothing(emission_count, unique_word_count, change_tag) * transition_prob[tag][change_tag])
            if prob > max_probability:  # get max probability
                max_probability = prob
                back_pointer = temp_row

    matrix[c_r][c_c].probability = max_probability
    matrix[c_r][c_c].back_pointer = back_pointer
    return matrix


# I started doing back tracking. I went back to the back-pointer. With this process i found the predicted tags.
def back_tracking(start_r, col_count, matrix, guess_tag):
    guess_tag.append(matrix[start_r][col_count].r_tag)
    start_r = matrix[start_r][col_count].back_pointer
    col_count = col_count - 1
    if start_r > -1:
        back_tracking(start_r, col_count, matrix, guess_tag)
    return guess_tag


# I will calculated accuracy
def accuracy(test_tag_list,predict_tag_list):
    true_pre = 0
    total_tag = 0

    for i in range(len(test_tag_list)):
        for j in range(len(test_tag_list[i])):
            total_tag += 1
            if test_tag_list[i][j] == predict_tag_list[i][j]:
                true_pre += 1

    print((true_pre / total_tag) * 100)


def main():
    train_path = "train.txt"
    test_path = "test.txt"

    train_sentences = dataset(train_path)
    test_sentences = dataset(test_path)

    # hmm function call and return initial prob, transition prob, emission prob, emission_count, unique_word_count
    initial_prob, transition_prob, emission_prob, emission_count, unique_word_count = hidden_markov_model(train_sentences)

    test_sent_list = []
    test_tag_list = []
    predict_tag_list = []

    for sent in test_sentences:
        word = []
        tag = []
        for i in range(len(sent)):
            word.append(sent[i].split("~")[0].lower())
            tag.append(sent[i].split("~")[1])
        test_sent_list.append(word)
        test_tag_list.append(tag)

    for test_sent in test_sent_list:
        col_count = len(test_sent)
        row_count = len(initial_prob.keys())

        matrix = [[MatrixItem for x in range(col_count)] for y in range(row_count)]
        # doing matrix tag number x word number
        for r in range(row_count):
            for c in range(col_count):
                matrix[r][c] = MatrixItem(list(initial_prob.keys())[r], test_sent[c], 0.0, 0)
        # matrix = define_matrix(col_count, row_count, initial_prob, test_sent)

        # apply viterbi each matrix item
        for current_column in range(col_count):
            for current_row in range(row_count):
                matrix = viterbi(current_row, current_column, matrix, initial_prob,
                                 transition_prob, emission_prob, emission_count, unique_word_count)

        # In the last column (so word)  we find the row with the highest probability, namely tag (row) .
        guess_tags = []
        max_value = 0
        start_r = 0
        for current_row in range(row_count):
            new_val = matrix[current_row][col_count - 1].probability
            if new_val > max_value:
                max_value = new_val
                start_r = current_row

        # doing back-tracking
        back_tracking(start_r, col_count-1, matrix, guess_tags)
        guess_tags.reverse()
        predict_tag_list.append(guess_tags)

    # print(test_tag_list)
    # print(predict_tag_list)

    # call accuracy function
    accuracy(test_tag_list, predict_tag_list)


if __name__ == '__main__':
    main()
