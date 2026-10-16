```@meta
CurrentModule = LiftAndLearn 
```

```@contents
Pages = ["api.md"]
```

# API
All APIs of LiftAndLearn that are not already documented in the manual pages, listed in an
unstructured manner.

```@autodocs
Modules = [LiftAndLearn]
Order = [:module, :type, :function, :macro]
Filter = t -> !(t in Main.MANUAL_DOCUMENTED)
```
