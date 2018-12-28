# coding=utf-8
import math
import random


class HMM:
    def __init__(self, train_file_path):
        self.file_path = train_file_path
        self.pi = {}                    # 初始状态概率分布
        self.A = {}                     # 状态转移概率矩阵
        self.B = {}                     # 观察值概率矩阵
        self.pos = []                   # 所有词性
        self.tag_fre = {}               # 词性出现频率
        self.A_class = {}               # 状态转移平滑处理矩阵
        self.B_class = {}               # 观察概率平滑处理矩阵

    def build_hmm(self):
        all_words = set()
        with open(self.file_path, 'r', encoding='utf-8') as f:
            all_sentence = f.readlines()
        line_counts = len(all_sentence)
        for sen in all_sentence:
            sen = sen.rstrip('\n')
            word_count = sen.split(" ")
            for word_with_tag in word_count:
                words = word_with_tag.split('/')
                if len(words) == 2:
                    all_words.add(words[0])
                    if words[1] not in self.pos:
                        self.pos.append(words[1])

        print("first step done")
        print(len(self.pos))

        for tag in self.pos:
            self.pi[tag] = 0
            self.tag_fre[tag] = 0
            self.A[tag] = {}
            self.B[tag] = {}
            for next_tag in self.pos:
                self.A[tag][next_tag] = 0
            for word in all_words:
                self.B[tag][word] = 0

        for sen in all_sentence:
            sen = sen.rstrip('\n')
            tmp_word = sen.split(" ")
            counts = len(tmp_word)
            head_word = tmp_word[0].split('/')
            self.pi[head_word[1]] += 1
            self.tag_fre[head_word[1]] += 1
            self.B[head_word[1]][head_word[0]] += 1
            for i in range(1, counts):
                current_word = tmp_word[i].split('/')
                pre_word = tmp_word[i - 1].split('/')
                if len(current_word) == 2:
                    if current_word[-1] and pre_word[-1]:
                        self.tag_fre[current_word[-1]] += 1
                        self.A[pre_word[-1]][current_word[-1]] += 1
                        self.B[current_word[-1]][current_word[0]] += 1

        print("second step done")

        # 矩阵中零太多，做平滑处理
        for tag in self.pos:
            self.A_class[tag] = 0
            self.B_class[tag] = 0
            if self.pi[tag] == 0:
                self.pi[tag] = 0.5 / line_counts
            else:
                self.pi[tag] = self.pi[tag] * 1.0 / line_counts
            for next_tag in self.pos:
                if self.A[tag][next_tag] == 0:
                    self.A_class[tag] += 1
                    self.A[tag][next_tag] = 0.5
            for word in all_words:
                if self.B[tag][word] == 0:
                    self.B_class[tag] += 1
                    self.B[tag][word] = 0.5

        for tag in self.pos:
            for next_tag in self.pos:
                self.A[tag][next_tag] = self.A[tag][next_tag] * 1.0 / (self.tag_fre[tag] + self.A_class[tag])
            for word in all_words:
                self.B[tag][word] = self.B[tag][word] * 1.0 / (self.tag_fre[tag] + self.B_class[tag])

        print("build done")

    def predict_pos_tags(self, test_file_path, truth_file_path):
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_sentence = f.readlines()

        with open(truth_file_path, 'r', encoding='utf-8') as f:
            truth_tags = f.readlines()

        with open('./data/result.txt', 'w', encoding='utf-8') as f:
            total_accu = 0
            for ind in range(len(test_sentence)):
                res = []
                sen = test_sentence[ind]
                sen = sen.rstrip('\n').split(" ")
                sen = sen[:-1]
                sen_length = len(sen)
                delta = [{} for i in range(sen_length)]
                psi = [{} for i in range(sen_length)]
                for tag in self.pos:
                    for index in range(sen_length):
                        delta[index][tag] = -1e100
                        psi[index][tag] = ""

                for tag in self.pos:
                    if sen[0] in self.B[tag]:
                        delta[0][tag] = math.log(self.pi[tag] * self.B[tag][sen[0]])
                    else:
                        delta[0][tag] = math.log(self.pi[tag] * 0.5 / (self.B_class[tag] + self.tag_fre[tag]))

                for i in range(1, sen_length):
                    for tag in self.pos:
                        if sen[i] in self.B[tag]:
                            tmp = math.log(self.B[tag][sen[i]])
                        else:
                            tmp = math.log(0.5 / (self.B_class[tag] + self.tag_fre[tag]))
                        for pre_tag in self.pos:
                            if delta[i][tag] < (delta[i - 1][pre_tag] + math.log(self.A[pre_tag][tag]) + tmp):
                                delta[i][tag] = delta[i - 1][pre_tag] + math.log(self.A[pre_tag][tag]) + tmp
                                psi[i][tag] = pre_tag

                max_end = self.pos[0]
                for tag in self.pos:
                    if delta[sen_length - 1][max_end] < delta[sen_length - 1][tag]:
                        max_end = tag

                i = sen_length - 1
                res.append(max_end)
                while i > 0:
                    max_end = psi[i][max_end]
                    res.append(max_end)
                    i -= 1
                res.reverse()

                truth = truth_tags[ind].rstrip("\n").split(" ")
                truth = truth[:-1]
                if len(truth) != len(res):
                    print("预测与结果词数不符，此句有误")
                    continue

                word_count = len(truth)
                correct_num = 0
                for j in range(word_count):
                    if truth[j] == res[j]:
                        correct_num += 1

                accu = correct_num / word_count

                f.write('expected: ' + ' '.join(truth) + '\n')
                f.write('got: ' + ' '.join(res) + '\n')
                f.write('accu: ' + str(accu) + '\n\n')
                f.flush()

                print("第%d个句子准确率为：%f" % (ind + 1, accu))

                total_accu += accu

            average_accu = total_accu / len(test_sentence)
            print("平均准确率为：%f" % average_accu)
            f.write("平均准确率为：" + str(average_accu))


if __name__ == '__main__':
    hmm = HMM('./data/simple_train_raw_data.txt')
    hmm.build_hmm()
    hmm.predict_pos_tags('./data/simple_test_words_data.txt', './data/simple_test_tags_data.txt')
