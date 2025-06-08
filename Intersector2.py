import pybedtools
import argparse # module for command line inputs

def main():
	parser = argparse.ArgumentParser(description="Input two bed files and this function quantifies the overlap between states for each bed file. Designed for a masters project.")
	parser.add_argument("input_file_a", type=str, help="Path to the hmm bed file")
	parser.add_argument("input_file_b", type=str, help="Path to the annotation bed file")
	args = parser.parse_args()
	
	a = pybedtools.BedTool(args.input_file_a)
	b = pybedtools.BedTool(args.input_file_b)
	
	original_state_counts = {}
	for interval in a:
		state = interval[3]
		original_state_counts[state] = original_state_counts.get(state, 0) + 1
	intersect = a.intersect(b, wa = True)	
	
	intersect_state_counts = {}
	for interval in intersect:
		state = interval[3]
		intersect_state_counts[state] = intersect_state_counts.get(state, 0) + 1
		
	print(intersect_state_counts)
	print(original_state_counts)
	
if __name__ == "__main__":
    main()
