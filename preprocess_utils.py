# Copyright (c) 2022 Yipeng Zhang. Licensed under the BSD 3-clause license.

import re, json
from transformers.pytorch_transformers import BertTokenizer


###############################################################
###### AMR preprocessing
###############################################################

'''
Input
- raw_amr (str): original AMR string (which might contain newlines and '/', etc.) for one sentence
- substitute_list (Dict[str, str]): a dictionary that maps some AMR role strings to the actual strings that are passed to the tokenizer
Output
- cleaned_amr (str): linearized and preprocessed AMR string ready to be passed to the tokenizer
'''
def clean_amr_str(raw_amr, substitute_list):
	linearized_amr = ' '.join(raw_amr.replace('\n', '').split())
	linearized_amr = linearized_amr.replace(' /', '').replace('(', '( ').replace(')', ' )') 

	for key, value in substitute_list.items():
		linearized_amr = linearized_amr.replace(key, value)

	for s_i in range(40, -1, -1):
		linearized_amr = linearized_amr.replace(':op'+str(s_i), ':op').replace(':snt'+str(s_i), ':snt')

	split_amr = linearized_amr.split(' ')
	for s_i, s in enumerate(split_amr):
		if re.match(r'-[0-9][0-9]', s[-3:]):
			split_amr[s_i] = s[:-3]
	cleaned_amr = ' '.join(split_amr)

	return cleaned_amr



###############################################################
###### Generate node and edge token index mapping from tokenizer and preprocessed AMR
###### This part of the code is specific to the BERT Tokenizer used in our project
###############################################################

def _get_amr_token_map(tokens):
	parents = []

	node_indices = []
	edge_indices = []	 
	edges = {}	

	seen_left_paren = False
	z_idx = -1
	edge_idx = -1
	for i, token in enumerate(tokens):
		
		if token == '(':
			seen_left_paren = True
			if edge_idx != -1:
				edge_indices.append(tuple(range(edge_idx, i)))
				edge_idx = -1
		
		elif token == 'z' and tokens[i+1].startswith('##'):
			z_idx = i
			if edge_idx != -1:
				edge_indices.append(tuple(range(edge_idx, i)))
				edge_idx = -1
			
		elif token.startswith(':'):
			if z_idx != -1:
				node_indices.append(tuple(range(z_idx, i)))
				z_idx = -1
				
				if len(edge_indices) > 0:
					edges[len(edge_indices)-1] = (parents[-1], len(node_indices)-1)
				
			elif edge_idx != -1:
				end = edge_idx + 2 if tokens[edge_idx+1] == '-of' else edge_idx + 1
				edge_indices.append(tuple(range(edge_idx, end)))
				node_indices.append(tuple(range(end, i)))
				
				edges[len(edge_indices)-1] = (parents[-1], len(node_indices)-1)
				
			if seen_left_paren:
				parents.append(len(node_indices)-1)
				seen_left_paren = False
				
			edge_idx = i
			
		elif token == ')':
			if z_idx != -1:
				node_indices.append(tuple(range(z_idx, i)))
				z_idx = -1
				
				if len(edge_indices) > 0:
					edges[len(edge_indices)-1] = (parents[-1], len(node_indices)-1)
				
			elif edge_idx != -1:
				end = edge_idx + 2 if tokens[edge_idx+1] == '-of' else edge_idx + 1
				edge_indices.append(tuple(range(edge_idx, end)))
				node_indices.append(tuple(range(end, i)))
				
				edges[len(edge_indices)-1] = (parents[-1], len(node_indices)-1)
			
			edge_idx = -1
			
			if not seen_left_paren:
				parents.pop()
			seen_left_paren = False

	return node_indices, edge_indices, edges

def _remove_duplicates(tokens, node_indices, edge_indices, edges):
	duplicates = {}
	for i, node_i in enumerate(node_indices.copy()):
		for j, node_j in enumerate(node_indices.copy()):
			if i >= j:
				continue
			if tokens[node_i[0]] == 'z' and tokens[node_i[0]+1].startswith('##') and tokens[node_i[0]+1] == tokens[node_j[0]+1]:
				ent = tokens[node_i[0]]+tokens[node_i[0]+1][2:]
				if ent in duplicates:
					duplicates[ent].append(j)
				else:
					duplicates[ent] = [i, j]

	for z, dup in duplicates.items():
		dup_set = set(dup)
		merged_node = []
		for node in dup_set:
			merged_node.extend(node_indices[node])
		for node in dup_set:
			node_indices[node] = tuple(merged_node)

	for e in edges.keys():
		edges[e] = (node_indices[edges[e][0]], node_indices[edges[e][1]])

	node_indices = list(set(node_indices))

	for e in edges.keys():
		edges[e] = (node_indices.index(edges[e][0]), node_indices.index(edges[e][1]))

	return node_indices, edge_indices, edges

'''
Input
- tokenizer (BertTokenizer)
- cleaned_amr (str): preprocessed AMR string for one sentence
- substitute_list (Dict[str, str]): a dictionary that maps some AMR role strings to the actual strings that are passed to the tokenizer
Output
- tokens (List[str]): list of string tokens that are mapped back to the original roles
- node_indices (List[int]): each element in node_indices is a list of token indices that form an AMR node
- edge_indices (List[int]): each element in edge_indices is a list of token indices that form an AMR edge
- edges (Dict[int, Tuple[int, int]]): each element in edges is in the form of 'e: [n1, n2]' which forms a KE tuple, 
	where e is the index of the edge in edge_indices and n1, n2 are indices of the 2 connected nodes in node_indices
'''
def get_node_edge_indices(tokenizer, cleaned_amr, substitute_list):
	tokens = tokenizer.tokenize(cleaned_amr)
	node_indices, edge_indices, edges = _remove_duplicates(tokens, *_get_amr_token_map(tokens))

	substitute_list_reversed = {v: k for k, v in substitute_list.items()}
	for tok_i, token in enumerate(tokens):
		if token in substitute_list_reversed:
			tokens[tok_i] = substitute_list_reversed[token]
	return tokens, node_indices, edge_indices, edges


###############################################################
###### Generate token index mapping (from each object tag back to each object region) from tokenizer and object detector outputs
###### In Oscar+ each region ocupies one token
###### There could be several tag tokens that correspond to the same region, but not vice versa
###### This part of the code is specific to the BERT Tokenizer used in our project
###############################################################

'''
Input
- tokenizer (BertTokenizer)
- object_tags (List[str]): list of object tags extracted by some pretrained object detector
Output
- mapping (List[int]): mapping[i] is the region index that corresponds to the i-th object tag
'''
def get_tag2region_token_map(tokenizer, object_tags):
	mapping = []
		
	tokens = tokenizer.tokenize(' '.join(object_tags))
	object_tags = [s.replace(' ', '') for s in object_tags]

	obj_idx = 0
	curr_obj = ''
	for token in tokens:
		mapping.append(obj_idx)

		if token[0] == '#':
			token = token[2:]
		curr_obj += token

		if curr_obj == object_tags[obj_idx]:
			curr_obj = ''
			obj_idx += 1

	return mapping


if __name__ == "__main__":
	substitute_list = json.load(open('./data/amr_substitute.json', 'r'))
	tokenizer = BertTokenizer.from_pretrained('/home/yipeng/vinvl-ws/models/vte/linear', do_lower_case=True)

	# Add AMR tokens
	with open('./data/amr_vocab.txt', 'r') as f:
		new_tokens = f.read().split('\n')
	while '' in new_tokens:
		new_tokens.remove('')
	if new_tokens:
		tokenizer.add_tokens(new_tokens)

	with open('./data/amr_special_tokens.txt', 'r') as f:
		new_special_tokens = f.read().split('\n')
	while '' in new_special_tokens:
		new_special_tokens.remove('')
	if new_special_tokens:
		tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

	# A test example from the FGVE training set
	test_amr = '(z0 / hold-01\n      :ARG0 (z1 / woman\n            :mod (z2 / continent\n                  :name (z3 / name\n                        :op1 "Africa"))\n            :mod (z4 / young))\n      :ARG1 (z5 / octopus\n            :ARG1-of (z6 / string-01))\n      :time (z7 / date-entity\n            :dayperiod (z8 / night)))'

	print('Raw AMR:\n{}\n'.format(test_amr))
	cleaned_amr = clean_amr_str(test_amr, substitute_list)
	print('Cleaned AMR:\n{}\n'.format(cleaned_amr))

	print('-----------------------')

	tokens, node_indices, edge_indices, edges = get_node_edge_indices(tokenizer, cleaned_amr, substitute_list)
	node_strs = [' '.join([tokens[tok_i] for tok_i in node_i]) for node_i in node_indices]
	edge_strs = [' '.join([tokens[tok_i] for tok_i in edge_i]) for edge_i in edge_indices]
	tuple_strs = ['{} {} {}'.format(node_strs[node_i], edge_strs[edge_i], node_strs[node_j]) for edge_i, (node_i, node_j) in edges.items()]
	print('Tokens:\n{}\n'.format(tokens))
	print('Nodes:\n{}\n'.format('\n'.join(node_strs)))
	print('Edges:\n{}\n'.format('\n'.join(edge_strs)))
	print('Tuples:\n{}\n'.format('\n'.join(tuple_strs)))

	print('-----------------------')

	test_objects = ['ship', 'black bear', 'big blue whale', 'cloud']
	tag2region_mapping = get_tag2region_token_map(tokenizer, test_objects)
	print('Objects:\n{}\n'.format(test_objects))
	print('Tag to region mapping:\n{}\n'.format(tag2region_mapping))

