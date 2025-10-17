#!/usr/bin/env python3
import csv
import re
import sys
import os

def parse_input_file(input_file_path, output_file_path):
    """
    Process a text file based on "Simulation Run" marker lines:
    - The line before "Simulation Run" contains values for columns 1-11
    - The line after "Simulation Run" becomes column 12
    """
    # First check if input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file '{input_file_path}' does not exist.")
        return False
        
    print(f"Processing file: {input_file_path}")
    
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
            print(f"Read {len(lines)} lines from input file")
            
            rows = []
            simulation_count = 0
            
            # Find all "Simulation Run" lines
            for i in range(1, len(lines) - 1):  # Skip first and last lines as boundary check
                if "Simulation Run" in lines[i]:
                    simulation_count += 1
                    print(f"Found Simulation Run marker at line {i+1}")
                    
                    # Get line before Simulation Run
                    before_line = lines[i-1].strip()
                    print(f"  Line before: {before_line}")
                    
                    # Extract numbers from line before
                    numbers = re.findall(r'-?\d+\.?\d*', before_line)
                    print(f"  Found {len(numbers)} numbers in line before")
                    
                    if len(numbers) < 11:
                        print(f"  Warning: Line {i} has only {len(numbers)} values, expected 11")
                        # Pad with empty strings if needed
                        numbers = numbers + [''] * (11 - len(numbers))
                    elif len(numbers) > 11:
                        print(f"  Note: Line {i} has {len(numbers)} values, using first 11")
                        numbers = numbers[:11]
                    
                    # Get line after Simulation Run
                    after_line = lines[i+1].strip()
                    print(f"  Line after: {after_line}")
                    
                    # Create a row with the 11 numbers from before line and value from after line as 12th column
                    row = numbers[:11]
                    row.append(after_line)
                    
                    rows.append(row)
            
            print(f"Found {simulation_count} Simulation Run markers")
            
            if not rows:
                print("No valid data rows found to write to CSV")
                return False
                
            # Write to CSV
            try:
                with open(output_file_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    # Write header
                    header = [f'col{i+1}' for i in range(12)]  # 12 columns total
                    csv_writer.writerow(header)
                    # Write data rows
                    csv_writer.writerows(rows)
                    
                print(f"Successfully processed {len(rows)} rows to {output_file_path}")
                return True
            except Exception as e:
                print(f"Error writing to CSV file: {e}")
                return False
                
    except Exception as e:
        print(f"Error processing input file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python parse_3_lines_to_csv.py input_file.txt output_file.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Starting processing of {input_file} to {output_file}")
    success = parse_input_file(input_file, output_file)
    
    if success:
        print("Processing completed successfully")
    else:
        print("Processing failed")
        sys.exit(1)