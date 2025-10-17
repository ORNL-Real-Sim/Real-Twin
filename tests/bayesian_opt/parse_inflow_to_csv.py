import re
import csv
import sys

def extract_solution_and_metrics(file_path):
    data = []
    current_solution = None
    mean_gen = None
    geh_below_5_percent = None
    
    lines_read = 0
    solution_count = 0
    
    print(f"Starting to process file: {file_path}")
    
    try:
        with open(file_path, 'r') as file:
            content = file.readlines()
            total_lines = len(content)
            print(f"File contains {total_lines} lines")
            
            for line_number, line in enumerate(content, 1):
                lines_read += 1
                line = line.strip()
                
                if lines_read % 1000 == 0:
                    print(f"Processed {lines_read} lines...")
                
                # Check if this line contains a solution
                if "Solution Under Investigation" in line:
                    solution_count += 1
                    print(f"Found solution marker #{solution_count} at line {line_number}: {line[:100]}...")
                    
                    # If we already have a solution and its metrics, add them to our data
                    if current_solution is not None and mean_gen is not None and geh_below_5_percent is not None:
                        objective = max(0, mean_gen - 3) + max(0, 85 - geh_below_5_percent)
                        data.append({
                            'solution': current_solution,
                            'mean_gen': mean_gen,
                            'geh_below_5_percent': geh_below_5_percent,
                            'objective': objective
                        })
                        print(f"Added solution to dataset: mean_gen={mean_gen}, geh_below_5_percent={geh_below_5_percent}")
                    
                    # Extract the new solution
                    match = re.search(r'Solution Under Investigation\s*\[(.*?)\]', line)
                    if match:
                        current_solution = match.group(1)
                        print(f"Extracted solution values: {current_solution[:50]}...")
                    else:
                        print(f"WARNING: Could not extract solution values from line: {line[:100]}...")
                        current_solution = "unknown"
                    
                    # Reset metrics for new solution
                    mean_gen = None
                    geh_below_5_percent = None
                
                # Check if this line contains mean_gen
                elif "mean_gen" in line and current_solution is not None:
                    match = re.search(r'mean_gen\s+([\d.]+)', line)
                    if match:
                        try:
                            mean_gen = float(match.group(1))
                            print(f"Found mean_gen: {mean_gen}")
                        except ValueError:
                            print(f"WARNING: Could not convert mean_gen to float: {match.group(1)}")
                
                # Check if this line contains geh_below_5_percent
                elif "geh_below_5_percent" in line and current_solution is not None:
                    match = re.search(r'geh_below_5_percent\s+([\d.]+)', line)
                    if match:
                        try:
                            geh_below_5_percent = float(match.group(1))
                            print(f"Found geh_below_5_percent: {geh_below_5_percent}")
                        except ValueError:
                            print(f"WARNING: Could not convert geh_below_5_percent to float: {match.group(1)}")
    
        print(f"Finished reading file. Read {lines_read} lines.")
        print(f"Found {solution_count} 'Solution Under Investigation' markers.")
        
        # Don't forget to add the last solution if it exists
        if current_solution is not None and mean_gen is not None and geh_below_5_percent is not None:
            objective = max(0, mean_gen - 3) + max(0, 85 - geh_below_5_percent)
            data.append({
                'solution': current_solution,
                'mean_gen': mean_gen,
                'geh_below_5_percent': geh_below_5_percent,
                'objective': objective
            })
            print(f"Added final solution to dataset: mean_gen={mean_gen}, geh_below_5_percent={geh_below_5_percent}")
    
    except Exception as e:
        print(f"Error while processing file: {e}")
    
    print(f"Extracted {len(data)} complete data rows")
    return data

def write_to_csv(data, output_file):
    if not data:
        print("No data found to write to CSV.")
        return False
    
    fieldnames = ['solution', 'mean_gen', 'geh_below_5_percent', 'objective']
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Successfully wrote {len(data)} solutions to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing CSV: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py input_file.txt output_file.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Processing {input_file} to create {output_file}")
    
    try:
        data = extract_solution_and_metrics(input_file)
        if data:
            success = write_to_csv(data, output_file)
            if success:
                print("Processing completed successfully.")
            else:
                print("Failed to write CSV file.")
                sys.exit(1)
        else:
            print("No valid data rows found to write to CSV.")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()