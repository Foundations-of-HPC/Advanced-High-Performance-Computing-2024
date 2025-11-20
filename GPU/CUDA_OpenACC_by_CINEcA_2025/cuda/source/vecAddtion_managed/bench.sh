#!/bin/bash

# benchmark.sh - Vector Addition Benchmark Automation and Summary Table

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
sizes=(1000000 2500000 5000000)

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

# Generate summary table
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Performance Summary${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Performance Summary Table:"
echo "=========================="
printf "%-25s | %15s | %10s | %15s\n" "Version" "Time" "Speedup" "Bandwidth"
echo "--------------------------|-----------------|------------|----------------"

# Re-run a single value of N for comparison of each mode
N=10000000
modes=("CPU Single Thread" "GPU Single Thread" "GPU Single Block" "GPU Multiple Blocks")

for mode_name in "${modes[@]}"; do
    temp_result=$(mktemp)
    if $EXECUTABLE $N > "$temp_result" 2>&1; then
        time_line=$(grep -E "Time: [0-9]+ ns" "$temp_result" | head -1)
        speedup_line=$(grep -E "Speedup: " "$temp_result" | head -1)
        bandwidth_line=$(grep -E "Bandwidth: " "$temp_result" | head -1)

        if [[ $time_line =~ ([0-9,]+)\ ns ]]; then time="${BASH_REMATCH[1]} ns"; else time="N/A"; fi
        if [[ $speedup_line =~ Speedup:\ ([0-9\.]+)x ]]; then speedup="${BASH_REMATCH[1]}"; else speedup="N/A"; fi
        if [[ $bandwidth_line =~ Bandwidth:\ ([0-9\.]+\ (MB\/s|GB\/s)) ]]; then bandwidth="${BASH_REMATCH[1]}"; else bandwidth="N/A"; fi

        printf "%-25s | %15s | %10sx | %15s\n" "$mode_name" "$time" "$speedup" "$bandwidth"
    else
        printf "%-25s | %15s | %10s | %15s\n" "$mode_name" "ERROR" "N/A" "N/A"
    fi
    rm -f "$temp_result"
done

