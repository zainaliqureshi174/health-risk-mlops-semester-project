# Model Evaluation Report

## Models Compared
1. Logistic Regression: 87.13% accuracy
2. Random Forest: 87.00% accuracy  
3. Federated Learning: 87.13% accuracy

## Key Findings
- Federated = Centralized performance
- Privacy preserved (69MB stayed local)
- Fast training (3 minutes, 5 rounds)

## Trade-offs
- **Accuracy vs Privacy:** No accuracy loss with FL
- **Speed vs Distributed:** Minimal overhead
- **Complexity vs Benefit:** Worth implementation

## Error Analysis
- Low F1 scores indicate class imbalance
- High accuracy but low recall for positive class
- Future: Use SMOTE or class weights
