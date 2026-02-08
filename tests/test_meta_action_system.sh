#!/bin/bash
# Quick test script for meta-action annotation system

set -e  # Exit on error

echo "=========================================="
echo "Meta-Action System Quick Test"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if egomotion data exists
EGOMOTION_DIR="/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar/labels/egomotion_corrected"

if [ ! -d "$EGOMOTION_DIR" ]; then
    echo -e "${YELLOW}Warning: Egomotion data not found at $EGOMOTION_DIR${NC}"
    echo "Please run tools/1_extract_egomotion.py first"
    exit 1
fi

echo -e "${GREEN}✓ Egomotion data found${NC}"

# Count available chunks
NUM_CHUNKS=$(ls -1 "$EGOMOTION_DIR"/egomotion.chunk_*.zip 2>/dev/null | wc -l)
echo "Found $NUM_CHUNKS chunk(s)"

# Test on first chunk
FIRST_CHUNK=$(ls -1 "$EGOMOTION_DIR"/egomotion.chunk_*.zip 2>/dev/null | head -1 | xargs basename | sed 's/egomotion\.//' | sed 's/\.zip//')

echo ""
echo "Testing on chunk: $FIRST_CHUNK"
echo ""

# Step 1: Run meta-action annotation
echo "=========================================="
echo "Step 1: Running meta-action annotation..."
echo "=========================================="
python3 tools/3_meta_action_annotation.py --chunks "$FIRST_CHUNK"

echo ""
echo -e "${GREEN}✓ Annotation complete${NC}"
echo ""

# Step 2: Visualize first video
echo "=========================================="
echo "Step 2: Generating visualization..."
echo "=========================================="

# Get first video UUID from the annotation
ANNOTATION_FILE="/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar/labels/meta_actions/meta_actions.$FIRST_CHUNK.json"

if [ -f "$ANNOTATION_FILE" ]; then
    FIRST_UUID=$(python3 -c "import json; data=json.load(open('$ANNOTATION_FILE')); print(data[0]['video_uuid'])" 2>/dev/null || echo "")

    if [ ! -z "$FIRST_UUID" ]; then
        echo "Visualizing video: $FIRST_UUID"
        python3 tools/visualize_meta_actions.py "$FIRST_CHUNK" "$FIRST_UUID"
        echo -e "${GREEN}✓ Visualization complete${NC}"
    else
        echo -e "${YELLOW}Warning: Could not extract video UUID${NC}"
    fi
else
    echo -e "${YELLOW}Warning: Annotation file not found${NC}"
fi

echo ""

# Step 3: Analyze statistics
echo "=========================================="
echo "Step 3: Analyzing statistics..."
echo "=========================================="
python3 tools/analyze_meta_actions.py --chunks "$FIRST_CHUNK" --plot

echo ""
echo -e "${GREEN}✓ Statistics analysis complete${NC}"
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - Annotation: $ANNOTATION_FILE"
echo "  - Statistics: /home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar/labels/meta_actions/meta_action_statistics.png"

if [ ! -z "$FIRST_UUID" ]; then
    echo "  - Video visualization: /home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar/labels/meta_actions/${FIRST_UUID}_meta_actions.png"
fi

echo ""
echo -e "${GREEN}✓ All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "  1. View the generated visualizations"
echo "  2. Adjust thresholds in config file if needed"
echo "  3. Run on full dataset: python3 tools/3_meta_action_annotation.py"
echo ""
