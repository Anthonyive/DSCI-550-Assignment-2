from lxml import etree
from lxml.html import fromstring
import os, tempfile, string, re

def TTR(path_to_xml, extract_content=False, threshold=3):
    with open(path_to_xml, 'r') as inFile:
        # Remove all script, remark tags and empty lines
        tree = fromstring(inFile.read())
        etree.strip_elements(tree, 'script', 'remark')
        tag_removed = etree.tostring(tree, pretty_print=True).decode('utf-8')
        all_removed = os.linesep.join([s for s in tag_removed.splitlines() if s])
        
        TTRArray = [None] * len(all_removed.splitlines())
        content = []

        for i, line in enumerate(all_removed.splitlines()):
            try:
                tr = fromstring(line)
            except etree.ParserError:
                # this case may caused by: there're only end tags in the line
                # one or many
                x = 0
            else:
                x = 0
                for ch in tr.text_content():
                    if ch in string.ascii_letters:
                        x += 1

            y = len(re.findall('<.*?>', line))
            if y == 0:
                TTRArray[i] = x
            else:
                TTRArray[i] = x/y

            if extract_content and TTRArray[i] > threshold:
                content.append(tr.text_content())

    content = '\n'.join(content)
    if extract_content:
        return TTRArray, content
    else:
        return TTRArray
            


    

if __name__ == '__main__':
    a,b = TTR('522.xml', extract_content=True)
    print(b)
