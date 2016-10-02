import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords="axes fraction",
                              xytext=center_pt, textcoords="axes fraction", va="center",
                              ha="center", bbox=node_type, arrowprops=arrow_args)


def get_num_leafs(my_tree):
    num_leafs = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):
    max_depth = 0
    first_str = my_tree.keys()[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1

        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def retrieve_tree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: 'no', 1: 'yes'}}}},
                    {'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                ]
    return listOfTrees[i]


def plot_mid_text(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    create_plot.ax1.text(xMid, yMid, txtString)


def plot_tree(myTree, parentPt, nodeTxt):
    numLeafs = get_num_leafs(myTree)
    get_tree_depth(myTree)
    firstStr = myTree.keys()[0]
    cntrPt = (plot_tree.xOff + (1.0 + float(numLeafs))/2.0/plot_tree.totalW,\
    plot_tree.yOff)
    plot_mid_text(cntrPt, parentPt, nodeTxt)
        # Plot child
    plot_node(firstStr, cntrPt, parentPt, decision_node)
    #value
    secondDict = myTree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plot_tree(secondDict[key],cntrPt,str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(secondDict[key], (plot_tree.xOff, plot_tree.yOff),
                    cntrPt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cntrPt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD


def create_plot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num_leafs(inTree))
    plot_tree.totalD = float(get_tree_depth(inTree))
    plot_tree.xOff = -0.5/plot_tree.totalW; plot_tree.yOff = 1.0;
    plot_tree(inTree, (0.5,1.0), '')
    plt.show()
