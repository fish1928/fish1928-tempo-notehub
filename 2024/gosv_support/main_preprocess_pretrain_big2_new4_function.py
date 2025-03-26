from collections import defaultdict
from copy import deepcopy
import sys

# abeabcdabcabc; abeabadabab
# so dedicated




def process_entitys(lines_all, last_n=None, window=None):
    mydict = defaultdict(list)
    i_current = 0
    list_i_previous = []
    i_next = False
    width_index = 0

    count_while = 0

    while True:
        count_while += 1
        if i_current >= len(lines_all):

            # copyed from anchor A，最后一个是重复的
            if list_i_previous:
                i_previous = list_i_previous[-1][0]

                if (i_previous + width_index) == (i_current - width_index):  # 如果发现连续
                    # print(f'jinyujinyu111 {i_previous + width_index + 1}: {i_current - width_index}, {i_previous + width_index} {i_current}')

                    lines_all = lines_all[:i_previous + width_index] + lines_all[
                                                                       i_current:]  # TODO: 这里删除的其实是上一个而不是这一个 [a0, a1, a2] -> [a0, a2, next], i_current在next

                    list_i_previous = []
                    i_next = False
                    i_current = i_previous + width_index  # TODO: 这里i_current更新到原来a2的位置

                    width_index = 0
                    # i_current += 1    #TODO: 这个要删除，为什么，因为删除的是上一个？
                    continue  # 清空并且继续
                # end

            break
        # end

        # if count_while % 10000 == 0:
        #     print(i_current, lines_all[i_current])
        # # end

        line_current = lines_all[i_current]
        # print(i_current, lines_all[i_current], list_i_previous, width_index, i_next, mydict, lines_all)

        if i_next:
            if not list_i_previous:
                #print('回退清空，往后走一格')
                mydict[line_current].append((i_current, line_current))
                i_current += 1
                i_next = False
                width_index = 0
                continue
            # end
        # end


        if not list_i_previous:
            #print('if not list_i_previous')
            if line_current not in mydict:
                mydict[line_current].append((i_current, line_current))
                i_current += 1
                continue
            else:
                list_i_previous = deepcopy(mydict[line_current])  # 启动模式
                if window:
                    list_i_previous = [i_p for i_p in list_i_previous if i_current - i_p[0] <= window]  # 发现15000行大循环
                # end
                # list_i_previous = [i_p for i_p in list_i_previous if i_current - i_p[0] <= 20000]             #发现15000行大循环
                # list_i_previous = list_i_previous[-2:]
                if last_n:
                    list_i_previous = list_i_previous[-last_n:]
                # end

                #TODO: 可能在这里
                if not list_i_previous:
                    mydict[line_current].append((i_current, line_current))
                    i_current += 1
                    continue
                else:
                    width_index += 1
                    i_current += 1
                    continue
                # end
            # end
        # end

        if list_i_previous:
            i_previous = list_i_previous[-1][0]

            if (i_previous + width_index) == (i_current - width_index):  # 如果发现连续
                #print(f'jinyujinyu111 {i_previous + width_index + 1}: {i_current - width_index}, {i_previous + width_index} {i_current}')

                lines_all = lines_all[:i_previous + width_index] + lines_all[i_current:]  # TODO: 这里删除的其实是上一个而不是这一个 [a0, a1, a2] -> [a0, a2, next], i_current在next

                list_i_previous = []
                i_next = False
                i_current = i_previous + width_index                                     #TODO: 这里i_current更新到原来a2的位置

                width_index = 0
                # i_current += 1    #TODO: 这个要删除，为什么，因为删除的是上一个？
                continue  # 清空并且继续
            # end

            if lines_all[i_previous + width_index] == lines_all[i_current]:   # 如果line old == line now
                #print('if lines_all[i_previous + width_index] == lines_all[i_current]:   # 如果line old == line now')

                # TODO: verify this
                #print('i_previous + width_index({}) : i_current - width_index({})'.format(i_previous + width_index, i_current - width_index))
                if (i_previous + width_index) + 1 == (i_current - width_index):     # 如果发现重复
                    #print('if i_previous + width_index >= i_current - width_index: {} {}  # 如果发现重复'.format(i_previous + width_index, i_current))
                    lines_all = lines_all[:i_previous + width_index] + lines_all[i_current:]  # 可能有问题

                    list_i_previous = []
                    i_next = False
                    i_current = i_previous + width_index

                    width_index = 0
                    i_current += 1  # TODO: 这个就不能删除，[a b c1 a b c2 a]，这里i_current在c2就发现重合了，所以去掉了中间的abc2后，i_current 被退回c1，所以要往后走一格
                    continue    # 清空并且继续
                else:       # 相同但是还没到
                    #print('else i_previous + width_index >= i_current - width_index:     # 没有发现重复')
                    width_index += 1
                    i_current += 1
                    continue
                # end


            else: # 不等于 -> 回退

                i_next = True
                #print('{} == {} 不等于 -> 回退到 {}'.format(lines_all[i_previous + width_index],  lines_all[i_current], i_current - width_index))
                i_current -= width_index
                width_index = 0
                list_i_previous.pop()
                # i_current += 1
                continue
            # end
        # end
    # end

    return lines_all
# end

###
def main(path_file_in, path_file_out):

    try:
        with open(path_file_in, 'r') as file:
            lines_all_raw = [line.lstrip().rstrip() for line in file.read().splitlines() if line]
        # end

        lines_all = [' '.join(process_entitys(line.split(' '), 3)) for line in lines_all_raw]
        lines_all = [line for line in lines_all if line.count(' ') > 0]
        lines_all = process_entitys(lines_all, 2, 20000)
        with open(path_file_out, 'w+') as file:
            file.write('\n'.join(lines_all))
        # end
    except Exception as ex:
        print('[ERROR] Fail to handle {}, reason: '.format(path_file_in), ex)
    # end
# end

# example:
if __name__ == '__main__':
    from datetime import datetime
    time1 = datetime.utcnow()
    main('data_test/__202307_SUCCESS_Ansible_Regression_Ubuntu_20_04_Desktop_ISO_70_2023_07_13_15_46_48_full_debug_log', 'output.txt')
    time2 = datetime.utcnow()
    print(time2 - time1)