# -*- coding: utf-8 -*-

# @File       : test.py
# @Date       : 2019-06-17
# @Author     : Jerold
# @Description: test2


R = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
R1 = ['a','c','e','g','i','k','l','m']
R2 = ['b','c','d','h','k']
R3 = ['d','f','g','n']
R4 = ['b','f','g','i','j']
R5 = ['b','k','n']

Rn = {'R1': R1,'R2': R2,'R3': R3,'R4': R4,'R5': R5}

C = ['b', 'd', 'f', 'l', 'n']

# 将数据转换成 元素 - 被哪些子集包含 的函数
def get_element2sets_dic(rn):
    # 定义存储元素被包含子集的字典
    elem_dict = {}
    # 依次遍历
    for r in rn:
        for ele in rn[r]:
            # 如果元素已经是一个key，则在该元素的被引用集合列表中增加该当前集合
            if ele in elem_dict:
                elem_dict[ele].append(r)
            # 如果元素还不是key，则建立此元素到字典中
            else:
                elem_dict[ele] = [r]

    return elem_dict

def test2():

    # 获取转换后的数据
    dic_elem2sets = get_element2sets_dic(rn=Rn)
    """
    # 打印出来看看
    for key, value in sorted(dic_elem2sets.items(), key=lambda x: x[0]):
        print(key, ':', value)
    """

    # 统计各个子集出现次数的字典
    R_count = {'R1': 0,'R2': 0,'R3': 0,'R4': 0,'R5': 0}

    # 创建目标集合C各元素对于的子集字典，即 dic_elem2sets 中包含目标集合C原色的一个子字典
    dict_C = dict((key, value.copy()) for key, value in dic_elem2sets.items() if key in C)

    """
    # 打印出来看看
    for key, value in dict_C.items():
        print(key, ':', value)
    """

    # 遍历来统计每个子集出现了多少次
    for key, value in dict_C.items():
        for r in value:
            R_count[r] += 1

    # 按照出现次数进行排序
    R_count = sorted(R_count.items(), key=lambda x: x[1])

    """ 打印出来看看
    print(R_count)
    """

    # 结果集合
    res = []
    # 从小到大来判断该集合是否不需要，不需要的标准是，是否剔除元素后会存在某个元素没有被包含集合了
    for r,count in R_count:
        # 出现过的才判断
        if count > 0:
            # 标识一个子集是否需要
            is_need = False
            for key, r_set in dict_C.items():
                # 如果某个元素的被包含子集中只有一个子集，且就是当前子集，则不可以剔除掉，需要当前子集
                if len(r_set) == 1 and r_set[0] == r:
                    is_need = True
                    break

            # 如果需要
            if is_need:
                # 在结果集中增加
                res.append(r)
            # 如果不需要，则从所有元素的集合中剔除此集合
            else:
                for key, r_set in dict_C.items():
                    if r in r_set:
                        r_set.remove(r)

    print(res)
    return res

if __name__ == '__main__':
    test2()