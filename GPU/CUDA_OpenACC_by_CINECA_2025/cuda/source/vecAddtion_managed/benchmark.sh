#!/bin/bash

# benchmark.sh - Vector Addition Benchmark Automation

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
EXECUTABLE="./vecAdd_managed.x"
OUTPUT_FILE="benchmark_results.txt"
TEMP_FILE="temp_output.txt"

if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: $EXECUTABLE not found. Please compile it first.${NC}"
    exit 1
fi

# Clean previous output
rm -f "$OUTPUT_FILE" "$TEMP_FILE"

# Vector sizes to benchmark (in elements)
sizes=(1000000 2500000 5000000 10000000 20000000 40000000)

for N in "${sizes[@]}"; do
    echo -e "${BLUE}Benchmarking N=$N...${NC}"

    if $EXECUTABLE $N > "$TEMP_FILE" 2>&1; then
        echo -e "${GREEN}✓ Completed for N=$N${NC}"
        echo "=== Vector Size: $N ===" >> "$OUTPUT_FILE"
        cat "$TEMP_FILE" >> "$OUTPUT_FILE"
        echo -e "\n" >> "$OUTPUT_FILE"
    else
        echo -e "${RED}✗ Failed for N=$N${NC}"
        cat "$TEMP_FILE"
    fi
done

echo -e "${YELLOW}Benchmarking complete. Results saved to $OUTPUT_FILE.${NC}"

