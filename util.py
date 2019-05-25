#UTILITY FUNCTIONS
maxCellWidth = 30
maxTableWidth = ((maxCellWidth * 2) + 3)

#make nice looking table
def dict_to_table(dictionary, title=""):
    res = "=" * maxTableWidth
    res += "\n"
    res += "| " + title + (" " * (maxTableWidth - 3 - len(title))) + "|"
    res += "\n"
    res += "=" * maxTableWidth
    res += "\n"
    for k, v in dictionary.items():
        kStr = str(k)
        vStr = str(v)
        res += "|" + kStr + (" " * (maxCellWidth - len(kStr))) + "|" + vStr + (" " * (maxCellWidth - len(vStr))) + "|"
        res += "\n"

    res += "=" * maxTableWidth
    res += "\n"
    return res
