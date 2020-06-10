import math
import re
import time

from nltk.corpus import stopwords

total_count = 0
index_map = {}
word_map = {}


def clean_text_tokens(text):
    text = re.sub(r'[\'\"\n]', '', text)
    text = re.sub(r'\.\s', r' ', text)
    tokens = re.split(r'[^a-zA-Z0-9\.\-]', text.lower())
    return tokens


def add_to_map(data_tokens):
    global total_count
    # data_tokens = re.split(r'[^a-zA-Z0-9\']', this_data)
    for word in data_tokens:
        if len(word) > 0:
            if word in word_map:
                val = word_map[word]
                word_map[word] = val + 1
                total_count += 1
            else:
                word_map[word] = 1
                total_count += 1


def process_data(data):
    global total_count

    for each in data:
        # add questions and answers to word_map
        add_to_map(clean_text_tokens(each[0]) + clean_text_tokens(each[1]))


def remove_stop_words(word_map):
    global total_count
    sw = stopwords.words('english')
    for each in sw:
        if re.search(r"\'", each):
            sw.remove(each)
            sw.append(re.sub(r"\'", r'', each))
    for each in sw:
        if each in word_map:
            val = word_map[each]
            total_count = total_count - val
            del (word_map[each])


def build_inverted_index(word_map, data):
    for idx in range(len(data)):
        q_tokens = clean_text_tokens(data[idx][0])
        a_tokens = clean_text_tokens(data[idx][1])
        tokens = q_tokens + q_tokens + a_tokens
        for word in tokens:
            if word in word_map:
                if word in index_map:
                    val = index_map[word]
                    if idx in val:
                        count = val[idx] + 1
                        val[idx] = count
                    else:
                        val[idx] = 1
                    index_map[word] = val
                else:
                    val = {idx: 1}
                    index_map[word] = val


def generate_tfidf(data, position_map, size):
    tfidf = [[0] * size for n in range(len(data))]
    i = -1
    for idx in range(len(data)):
        i += 1
        q_tokens = clean_text_tokens(data[idx][0])
        a_tokens = clean_text_tokens(data[idx][1])

        this_data = re.sub(r'[\'\"\n]', '', data[idx][0] + " " + data[idx][0] + " " + data[idx][1])
        this_data = re.sub(r'\.\s', r' ', this_data.lower())

        tokens = q_tokens + q_tokens + a_tokens
        sw = stopwords.words('english')
        for each in sw:
            if re.search(r"\'", each):
                sw.remove(each)
                sw.append(re.sub(r"\'", r'', each))
        for each in sw:
            if each in tokens:
                tokens.remove(each)
        for word in tokens:
            if word in position_map:
                count_word = this_data.count(word)
                tf = count_word / len(tokens)
                idf = math.log(len(data) / len(index_map[word]), 10)
                tfidf[i][position_map[word]] = tf * idf
    with open("tfidf.txt", 'a') as out:
        out.write(str(tfidf))
    return tfidf


def transform_query_tfidf(query, position_map, size):
    query_tfidf = [0] * size
    token_query = query.split(' ')
    sw = stopwords.words('english')
    for each in sw:
        if re.search(r"\'", each):
            sw.remove(each)
            sw.append(re.sub(r"\'", r'', each))
    for each in sw:
        if each in token_query:
            token_query.remove(each)
    for word in token_query:
        if word in position_map:
            count_word = query.count(word)
            tf = count_word / len(token_query)
            idf = math.log(data_len / len(index_map[word]), 10)
            query_tfidf[position_map[word]] = tf * idf
    return query_tfidf


def generate_results_t2(tfidf, query_tfidf):
    rank_map_t2 = {}
    j = 0
    for tfidf_row in tfidf:
        sum_val = 0
        tfidf_row_square = 0
        tfidf_query_square = 0
        # print("check")
        for i in range(0, len(tfidf_row)):
            sum_val = sum_val + tfidf_row[i] * query_tfidf[i]
            tfidf_row_square = tfidf_row_square + tfidf_row[i] * tfidf_row[i]
            tfidf_query_square = tfidf_query_square + query_tfidf[i] * query_tfidf[i]
        bottom = math.sqrt(tfidf_row_square * tfidf_query_square)
        if bottom == 0:
            bottom = 1
        rank_map_t2[j] = sum_val / bottom
        j += 1
    return sorted(rank_map_t2.items(), key=lambda x: x[1], reverse=True)


def process_results(query, tfidf, size):
    query_tfidf = transform_query_tfidf(query, position_map, size)
    start_time = time.time()
    result_2 = generate_results_t2(tfidf, query_tfidf)
    original_length_2 = 0
    for val in result_2:
        if val[1] > 0:
            original_length_2 += 1

    print("Results")
    print("*************")
    if len(result_2) > 10:
        result_2 = result_2[0:10]
    i = 1
    if original_length_2 == 0:
        print("No result found :(")
    for val in result_2:
        rank = val[0] + 1
        if val[1] > 0.0:
            print("Question: " + k[rank - 1][0])
            print("Rank-" + str(i) + ": doc " + str(rank) + " : " + str(val[1]))
        i += 1
    print("---------------")
    print("Total number of related documents " + str(original_length_2))
    print("---------------")


data_file = open(r"faq.txt", "r").read()

k = re.findall(r'\$\$\s*(.*)\n\^\^\s*(.*)', data_file)

data_len = len(k)

process_data(k)

remove_stop_words(word_map)

build_inverted_index(word_map, k)

top_tuple = sorted(word_map.items(), key=lambda x: x[1], reverse=True)
top_words = []
position_map = {}
matrix_size = len(word_map) if len(word_map) < 1000 else 1000
for i in range(matrix_size):
    key_word = top_tuple[i][0]
    top_words.append(key_word)
    position_map[key_word] = i

tfidf = generate_tfidf(k, position_map, matrix_size)

check = True
while check:
    query = input("please enter your search query...")
    process_results(query, tfidf, matrix_size)
    con = input("Do you want to continue : Y/N ?....")
    if con == "n" or con == "N":
        check = False
