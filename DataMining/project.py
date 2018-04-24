# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 10:37:22 2017

@author: Administrator
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint
import csv
import codecs

def count_tags(filename):
    elem_dict = {}
    for _, elem in ET.iterparse(filename, events=("start",)):
        if elem.tag in elem_dict:
            elem_dict[elem.tag] += 1
        else:
            elem_dict[elem.tag] = 1
    return elem_dict



lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


def key_type(element, keys):
    if element.tag == "tag":
        # YOUR CODE HERE
        k=element.get('k')
        if re.match(lower,k):
            keys["lower"] += 1
        elif lower_colon.match(k):
            keys["lower_colon"] += 1
        elif problemchars.search(k):
            keys["problemchars"] += 1
        else:
            keys["other"] += 1        
    return keys


def get_user(element):

    if "uid" in element.attrib:
        unique=element.get('uid')
        return unique
    

OSMFILE = "las-vegas_nevada.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "Ave":"Avenue",
            "Ave.":"Avenue",
            "Blvd":"Boulevard",
            "Blvd.":"Boulevard",
            "Dr":"Driver",
            "Ln":"Lane",
            "Ln.":"Lane",
            "Pkwy":"Parkway",
            "Rd":"Road",
            "Rd.":"Road",
            "Rd5":"Road #5",
            "St": "Street",
            "St.": "Street",
            "ave":"Avenue",
            "blvd":"Boulevard",
            "drive":"Drive",
            "parkway":"Parkway"
            }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):

    # YOUR CODE HERE
    for mapp in mapping:
        if name.find(mapp)>0:
            return (name.replace(mapp,mapping[mapp]))


OSM_PATH = "las-vegas_nevada.osm"        
NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
                             
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_tag(element,tag):
    tag={
        'id':element.attrib['id'],
        'key':tag.attrib['k'],
        'value':tag.attrib['v'],
        'type':'regular'
        }
    if LOWER_COLON.match(tag['key']):
        tag['type'], _, tag['key']=tag['key'].partition(':')
    return tag
    
def shape_way_node(element,i,nd):
    return {
        'id':element.attrib['id'],
        'node_id':nd.attrib['ref'],
        'position':i
        }
        

def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""
    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    # YOUR CODE HERE
    for tag in element.iter('tag'):
        if is_street_name(tag):
            m = street_type_re.search(tag.attrib['v'])
            if m:
                street_type = m.group()
                if street_type not in expected:
                    tag.attrib['v']=update_name(tag.attrib['v'], mapping)
                    print tag.attrib['v']
        new_tag = shape_tag(element,tag)
        if new_tag:
             tags.append(new_tag)
        
    if element.tag=='node':
        node_attribs={f:element.attrib[f] for f in node_attr_fields}
        return {'node':node_attribs,'node_tags':tags}
        
    elif element.tag=='way':
        way_attribs={f:element.attrib[f] for f in way_attr_fields}
        way_nodes=[shape_way_node(element,i,nd) for i,nd in enumerate(element.iter('nd'))]
        return {'way':way_attribs,'way_nodes':way_nodes,'way_tags':tags}

# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSM_PATH, validate=False)