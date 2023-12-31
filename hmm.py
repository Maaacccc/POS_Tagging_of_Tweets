import re
from spellchecker import SpellChecker
# ------------------------------------------------Q2a--------------------------------------------------------
def estimate_output_probabilities(training_file, delta):
    tag_counts = {}
    output_counts = {}
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                token, tag = line.strip().split('\t')
                token = token.lower()
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                output_counts.setdefault(tag, {})
                output_counts[tag][token] = output_counts[tag].get(token, 0) + 1

    curr_unique_words = []
    for tagDict in output_counts.values(): 
        curr_unique_words += tagDict.keys() 
    final_unique_words = set(curr_unique_words) 
    num_words = len(final_unique_words)

    output_probs = {}
    for tag, counts in output_counts.items(): 
        current_tag_count = tag_counts[tag] 
        output_probs[tag] = {} 
        for token, pairCount in counts.items(): 
            output_probs[tag][token] = (pairCount + delta) / (current_tag_count + delta * (num_words + 1)) 

    return output_probs

delta = 0.01
print(delta)
output_probs = estimate_output_probabilities('twitter_train.txt', delta)
with open('naive_output_probs.txt', 'w', encoding='utf-8') as f:
    for tag, probs in output_probs.items(): 
        for token, prob in probs.items():
            f.write(f"{tag}\t{token}\t{prob}\n")


# ------------------------------------------------Q4a-------------------------------------------------------
def compute_trans_probs(train_filename, delta):
    tag_counts = {}
    transition_counts = {}
    initial_counts = {}

    end_of_tweet = '</s>'

    with open(train_filename, 'r', encoding='utf-8') as f:
        previous_tag = '<s>'
        for line in f:
            if line.strip():
                token, tag = line.strip().split('\t')
                token = token.lower()

                # Increment the count of the current tag
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

                # If previous tag is <s> state, then increment the inital counts
                if previous_tag == '<s>':
                    initial_counts[tag] = initial_counts.get(tag, 0) + 1
                else:
                    transition_counts.setdefault(previous_tag, {})
                    transition_counts[previous_tag][tag] = transition_counts[previous_tag].get(tag, 0) + 1
                previous_tag = tag
            else:
                # Add a transition to the end of tweet tag
                transition_counts.setdefault(previous_tag, {})
                transition_counts[previous_tag][end_of_tweet] = transition_counts[previous_tag].get(end_of_tweet, 0) + 1

                # Reset the previous_tag when an empty line is encountered (end of a tweet)
                previous_tag = '<s>'
                        

    # Calculate the transition probabilities using MLE and smoothing
    transition_probs = {}
    initial_probs = {}

    for tag, count in initial_counts.items():
        initial_probs[tag] = (count + delta) / (sum(initial_counts.values()) + delta * (len(tag_counts) + 1))

    for prev_tag, counts in transition_counts.items():
        transition_probs[prev_tag] = {}
        total_transitions = sum(counts.values())

        for curr_tag, pair_count in counts.items():
            # Apply smoothing to the transition probability
            transition_probs[prev_tag][curr_tag] = (pair_count + delta) / (total_transitions + delta * (len(tag_counts) + 1)) 

    return initial_probs, transition_probs

initial_probs, transition_probs = compute_trans_probs('twitter_train.txt', delta)
with open('trans_probs.txt', 'w', encoding='utf-8') as f1, open('trans_probs2.txt', 'w', encoding='utf-8') as f2:
    # write the initial prob into both 'trans_probs.txt' and 'trans_probs2.txt'
    # note: whether writing initial prob into trans prob files or not has no effect on accuracies
    for tag, initial_prob in initial_probs.items(): 
        f1.write(f"<s>\t{tag}\t{initial_prob}\n")
        f2.write(f"<s>\t{tag}\t{initial_prob}\n")

    # write the transition prob into both 'trans_probs.txt' and 'trans_probs2.txt'
    for prev_tag, probs in transition_probs.items(): 
        for curr_tag, prob in probs.items():
            f1.write(f"{prev_tag}\t{curr_tag}\t{prob}\n")
            f2.write(f"{prev_tag}\t{curr_tag}\t{prob}\n")
    


def estimate_output_and_transition_probabilities(training_file, delta):
    # output prob, which will be written into the 'output_probs2.txt' as well
    output_probs = estimate_output_probabilities(training_file, delta)
    with open('output_probs.txt', 'w', encoding='utf-8') as f1, open('output_probs2.txt', 'w', encoding='utf-8') as f2:
        for tag, probs in output_probs.items():
            for token, prob in probs.items():
                f1.write(f"{tag}\t{token}\t{prob}\n")
                f2.write(f"{tag}\t{token}\t{prob}\n")
    
    # trans prob
    compute_trans_probs(training_file, delta)

# get the output for Q4a
estimate_output_and_transition_probabilities('twitter_train.txt', delta)

#  -----------------------------------------------Q4b-------------------------------------------------------
def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
     # Load output probabilities
    output_probs = {}
    with open(in_output_probs_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                tag, word, prob = line
                if tag not in output_probs:
                    output_probs[tag] = {}
                output_probs[tag][word] = float(prob)

     # Load transition probabilities
    transition_probs = {}
    with open(in_trans_probs_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                source_tag, dest_tag, prob = line
                if source_tag not in transition_probs:
                    transition_probs[source_tag] = {}
                transition_probs[source_tag][dest_tag] = float(prob)

    # Load tags from "twitter_tags.txt"
    with open(in_tags_filename, 'r', encoding='utf-8') as f:
        tags = f.read().strip().split('\n')

    # Define the nested viterbi function
    def viterbi(tokens):
        # number of words in the output sequence
        n = len(tokens)
        # number of possible tags at each time step (number of hidden states)
        T = len(tags)
        epsilon = 1e-10 

        # Initialize the trellis and backpointers matrices
        # These two are T x n matrices, each inner list represents a tag, and its elements correspond to the probabilities of that tag for each token in the input sequence.
        # 2D list, T rows, n cols
        trellis = [[0 for _ in range(n)] for _ in range(T)]
        backpointers = [[0 for _ in range(n)] for _ in range(T)]

        # Initialize the first column of the trellis, i is the row number                   
        for i, tag in enumerate(tags):
            trellis[i][0] = initial_probs.get(tag, epsilon) * output_probs.get(tag, {}).get(tokens[0], epsilon)

        # Iterate through the remaining tokens
        for t in range(1, n):
            for i, curr_tag in enumerate(tags):
                max_prob = -1
                max_idx = -1

                for j, prev_tag in enumerate(tags):
                    prob = (trellis[j][t-1]) \
                        * (transition_probs.get(prev_tag, {}).get(curr_tag, epsilon)) \
                        * (output_probs.get(curr_tag, {}).get(tokens[t], epsilon))

                    if prob > max_prob:
                        max_prob = prob
                        max_idx = j

                trellis[i][t] = max_prob
                backpointers[i][t] = max_idx

        # Find the index of the best final state
        best_final_idx = max(range(T), key=lambda i: trellis[i][-1] * (transition_probs.get(tags[i], {}).get('</s>', epsilon)))
            
        # Backtrack to find the best tag sequence
        best_tags = [tags[best_final_idx]]
        for t in reversed(range(1, n)):
            best_final_idx = backpointers[best_final_idx][t]
            best_tags.insert(0, tags[best_final_idx])

        return best_tags

    # Read untagged tweets, apply Viterbi algorithm, and write predictions to the file
    with open(in_test_filename, 'r', encoding='utf-8') as infile, open(out_predictions_filename, 'w', encoding='utf-8') as outfile:
        tokens = [] # like one sentence, before hitting a blank line
        for line in infile:
            if line.strip(): # if line not empty, one sentence not reaching end yet
                tokens.append(line.strip().lower())
            else: # reached end of sentence
                best_tags = viterbi(tokens) # find best tags for this one sentence only
                for tag in best_tags:
                    outfile.write(tag + '\n')
                outfile.write('\n') # a blank line before predictions for next sentence
                tokens = [] # reset to prep for next sentence

# ------------------------------------------------Q4c--------------------------------------------------------
'''
delta = 0.01
Viterbi prediction accuracy:   1083/1378 = 0.785921625544267

delta = 0.1
Viterbi prediction accuracy:   1080/1378 = 0.783744557329463

delta = 1
Viterbi prediction accuracy:   1078/1378 = 0.7822931785195936

delta = 10
Viterbi prediction accuracy:   1067/1378 = 0.7743105950653121
'''


# ------------------------------------------Q5a------------------------------------------------------------------
'''
Could your new POS tagger better handle unseen words? 
The new POS tagger can better handle unseen words by checking special conditions at all time steps (which equals to the length of each Tweet). 
1. If the token starts with “@”, it can be directly predicted to be a username, which corresponds to tag “@”. 
2. If the token starts with “http”, then it can be directly predicted to be a website link, which corresponds to tag “U”. 
3. We observed that all the numerical values, which include floats, integers, fractions, timings, are all associated with the tag ”$”. Hence, we can be sure that if the first character in a token is any number from 0 to 9, the token can be directly predicted to have the tag “$”. 
4. For tokens in the form of “#” followed by an integer, the tag associated with it will always be “$”. Hence, we will add this in our check. 
5. For tokens in the form of “$” followed by an integer, the tag associated with it will always be “$”. Hence, we will add this in our check.
6. To deal with typos made by the user, we use the Python package SpellChecker to replace the typos with the correct word. This preprocessing will happen before we pass the tweets to the viterbi algorithm to reduce the occurrence of typos being recognised as unseen words, thereby increasing the accuracy.
The abovementioned checks can avoid situations where these types of tokens that did not appear in the training data are recognised as unseen words in the test data, thereby increasing the accuracy.

Could your new POS tagger take advantage of linguistic patterns present in tweets? 
1. Elongated words
Elongated words appear frequently in Tweets as people write in informal language. Therefore, we can have a preprocessing function, which performs the following operations: Given a token, the repeated characters in the token will be replaced by their original form. We can create a regular expression pattern that matches any character that is repeated 2 or more times. We will also replace any matches of this pattern in the token with just a single occurrence of the character. For example, if token is "haaaappy", the preprocessed version of this token would be "happy", with the repeated "a" characters reduced to a single "a".
2. Emoticon 
Emoticons, which are pictorial representations of a facial expression created by combining keyboard characters, are another linguistic pattern in Twitter. We have researched a list of commonly used emoticons from a research paper (http://lnu.diva-portal.org/smash/get/diva2:1727896/FULLTEXT01.pdf) and included this list as a global variable, and check if the token is in the list. If yes, we can directly predict the token to be associated with tag “E”, which represents emoticons.
3. Hashtag 
“#” in Tweets is specifically used to join conversations about a particular topic or event. As such, any token that starts with “#” can be directly predicted to be associated with tag “#”.
4. Re-tweets
“RT” is a specially reserved word in Twitter, which is used when a user re-tweets a Tweet from another user. We observed that the RT is always associated with the tag “~”. As such, we can directly predict RT as “~” in our code.
'''



# ------------------------------------------Q5b----------------------------------------------------
# to deal with elongated words
def preprocess_word(word):
    # Replace repeated characters with their original form
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1", word)

# Initialise spell checker
spell = SpellChecker()

# to deal with typos
def spellcheck_word(word):
    # things to be spell checked: anything with first letter is a-z, A-Z, but not "rt"
    alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    if word[0] in alphabet_list and word != "rt" and word[:4] != "http" and word not in spell:        
        corrected_word = spell.correction(word)
        if corrected_word == None:
            return word
        else:
            return corrected_word    
    return word

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    # Load output probabilities
    output_probs = {}
    with open(in_output_probs_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                tag, word, prob = line
                if tag not in output_probs:
                    output_probs[tag] = {}
                output_probs[tag][word] = float(prob)

    # Load transition probabilities
    transition_probs = {}
    with open(in_trans_probs_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 3:
                source_tag, dest_tag, prob = line
                if source_tag not in transition_probs:
                    transition_probs[source_tag] = {}
                transition_probs[source_tag][dest_tag] = float(prob)

    # Load tags from "twitter_tags.txt"
    with open(in_tags_filename, 'r', encoding='utf-8') as f:
        tags = f.read().strip().split('\n')

    def viterbi(tokens):
        # number of words in the output sequence
        n = len(tokens)
        # number of possible tags at each time step (number of hidden states)
        T = len(tags)
        epsilon = 1e-10

        trellis = [[0 for _ in range(n)] for _ in range(T)]
        backpointers = [[0 for _ in range(n)] for _ in range(T)]

        emoticons = [':)', ':-)', ';)',
                     ';-)', ':(', ':P', '<3', ':/', ':o', 'xD', ':D', 'O_o']
        number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        for i, tag in enumerate(tags):
            if tag == '@' and tokens[0][0] == "@":
                trellis[0][0] = 1
            elif tag == '$' and len(tokens[0]) > 1 and tokens[0][0] == '#' and tokens[0][1] in number_list:
                trellis[9][0] = 1
            elif tag == '#' and tokens[0][0] == '#':
                trellis[24][0] = 1
            elif tag == 'U' and tokens[0][0:4] == 'http':
                trellis[21][0] = 1
            elif tag == '~' and tokens[0] == 'rt':
                trellis[3][0] = 1
            elif tag == 'E' and tokens[0] in emoticons:
                trellis[13][0] = 1
            elif tag == "$" and len(tokens[0]) > 1 and tokens[0][0] == '$' and tokens[0][1] in number_list:
                trellis[9][0] = 1
            elif tag == '$' and tokens[0][0] in number_list:
                trellis[9][0] = 1
            else:
                trellis[i][0] = initial_probs.get(
                    tag, epsilon) * output_probs.get(tag, {}).get(tokens[0], epsilon)

        # Iterate through the remaining tokens
        for t in range(1, n):
            for i, curr_tag in enumerate(tags):
                max_prob = -1
                max_idx = -1
                for j, prev_tag in enumerate(tags):
                    if (curr_tag == '@' and tokens[t][0] == '@'):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '@' produces '@User..'
                    elif (curr_tag == '$' and len(tokens[t]) > 1 and tokens[t][0] == '#' and tokens[t][1] in number_list):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '$' produces '#1'
                    elif (curr_tag == '#' and tokens[t][0] == '#'):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '#' produces '#..'
                    elif (curr_tag == 'U' and tokens[t][0:4] == 'http'):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag 'U' produces 'a website'
                    elif (curr_tag == '~' and tokens[t] == 'rt'):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '~' produces twitter nuances 'rt'
                    elif (curr_tag == 'E' and tokens[t] in emoticons):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag 'E' produces emoticons
                    elif (curr_tag == '$' and len(tokens[t]) > 1 and tokens[t][0] == '$' and tokens[t][1] in number_list):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '$' produces $1/ $2
                    elif (curr_tag == '$' and tokens[t][0] in number_list):
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * 1  # prob that curr_tag '$' produces numbers
                    else:
                        prob = (trellis[j][t-1]) * (transition_probs.get(prev_tag, {}).get(
                            curr_tag, epsilon)) * (output_probs.get(curr_tag, {}).get(tokens[t], epsilon))

                    if prob > max_prob:
                        max_prob = prob
                        max_idx = j

                trellis[i][t] = max_prob
                backpointers[i][t] = max_idx

        # Find the index of the best final state
        best_final_idx = max(range(T), key=lambda i: trellis[i][-1] * transition_probs.get(tags[i], {}).get('</s>', epsilon))

        # Backtrack to find the best tag sequence
        best_tags = [tags[best_final_idx]]
        for t in reversed(range(1, n)):
            best_final_idx = backpointers[best_final_idx][t]
            best_tags.insert(0, tags[best_final_idx])

        return best_tags

    with open(in_test_filename, 'r', encoding='utf-8') as infile, open(out_predictions_filename, 'w', encoding='utf-8') as outfile:
        tokens = []
        for line in infile:
            if line.strip():
                word = line.strip().lower()
                processed_word = preprocess_word(word)  # deal with elongated words, eg: isssssss -> is
                spell_checked_word = spellcheck_word(processed_word) # check typos
                tokens.append(spell_checked_word)              
            else:
                best_tags = viterbi(tokens)
                for tag in best_tags:
                    outfile.write(tag + '\n')
                outfile.write('\n')
                tokens = []

# ------------------------------------------------Q5c--------------------------------------------------------
'''
delta = 0.01
Viterbi2 prediction accuracy:  1137/1378 = 0.8251088534107403

delta = 0.1
Viterbi2 prediction accuracy:  1133/1378 = 0.8222060957910015

delta = 1
Viterbi2 prediction accuracy:  1130/1378 = 0.8200290275761973

delta = 10
Viterbi2 prediction accuracy:  1120/1378 = 0.8127721335268505
'''


# ---------------------------------------------decide the best delta value--------------------------------------------------------
'''
Looking across the accuracies in Q4c and Q5c, delta = 0.01 gives the highest accuracy for both Q4c and Q5c.
Hence, we will use delta = 0.01 for our submission. 
'''



def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = 'E:\Y2S2\BT3102_Computational Methods for Business Analytics\project\BT3102' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()