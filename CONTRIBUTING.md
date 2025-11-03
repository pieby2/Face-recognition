# Contributing

Thanks for your interest in contributing to this project. A few guidelines to help you get started:

1. Create issues for features or bugs before submitting a PR so we can discuss scope.
2. Fork the repo, create a feature branch, and open a pull request against `main`.
3. Keep changes small and focused. Add tests for new behavior where possible.
4. Run the synthetic evaluation locally to sanity-check matching logic:

```powershell
python -m FRS_Microservice.evaluation --n-identities 30 --n-queries 120 --match-ratio 0.5
```
