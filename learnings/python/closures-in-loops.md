# Learning: Closures in Loops

## The problem

When you create a function inside a `for` loop, the inner function captures the
*variable*, not its *value*. By the time the function runs, the loop variable
has already moved on to its final value.

```python
funcs = []
for x in ["a", "b", "c"]:
    def f():
        print(x)
    funcs.append(f)

funcs[0]()  # "c" — not "a"!
funcs[1]()  # "c"
funcs[2]()  # "c"
```

All three functions share the same `x` binding. After the loop, `x == "c"`,
so every function prints `"c"`.

## The fix: closure factory

Wrap the inner function in a factory that takes the value as a parameter.
Parameters are bound at call time, so each inner function gets its own copy:

```python
funcs = []
for x in ["a", "b", "c"]:
    def make_f(val):
        def f():
            print(val)
        return f
    funcs.append(make_f(x))

funcs[0]()  # "a"
funcs[1]()  # "b"
funcs[2]()  # "c"
```

## Where this shows up in the loop

`_make_on_update` in `loop.py` is a closure factory. During tool execution, we
iterate over tool calls and create an `on_update` callback for each one. Without
the factory, every callback would capture the *last* tool call's variables:

```python
# From loop.py
def _make_on_update(call_id, name, args):
    def on_update(partial):
        stream.push({
            "type": "tool_execution_update",
            "tool_call_id": call_id,
            "tool_name": name,
            "args": args,
            "partial": partial,
        })
    return on_update
```

Called once per iteration with the current values frozen into the closure.

## Alternative: default argument trick

A common shorthand that avoids the factory function:

```python
for x in ["a", "b", "c"]:
    def f(val=x):  # default arg evaluated at definition time
        print(val)
```

Works but is less readable — it looks like `f` takes an argument when it
doesn't really. The factory pattern is clearer about intent.
