# some boilerplate

```python
op = structured.GeneralizeOp(
    target=match(module_op, ["linalg.matmul"]),
)
structured.InterchangeOp(
    target=op.transformed,
    iterator_interchange=[2, 0, 1],
)
```

```python
# pipeline
with print_op_context(module_op, "pipeline"):
    theFor = match(
        module_op,
        ["scf.for"],
        # matched_op=transform.OperationType.get("scf.for"),
        op_attrs=DictAttr.get(
            {
                "__my_pipeline__": StringAttr.get("__my_pipeline__"),
            }
        ),
    )
    inner_for = match(
        theFor,
        ["scf.for"],
    )
    fors = transform.SplitHandleOp([any_op_t(), any_op_t()], inner_for)
    need_unroll = fors.results[0]
    # print_op(fors.results[0], "inner for", False)
    loop.LoopUnrollOp(need_unroll, factor=2)

transform.ApplyRegisteredPassOp(
    any_op_t(),
    match(module_op, ["gpu.func"]),
    "iree-codegen-gpu-pipelining",
)


loop.LoopPipelineOp(
    # transform.OperationType.get("scf.for"),
    any_op_t(),
    # match(module_op, ["scf.for"]),
    match(
        module_op,
        ["scf.for"],
        matched_op=transform.OperationType.get("scf.for"),
        op_attrs=DictAttr.get(
            {
                "__internal_linalg_transform__": StringAttr.get(
                    "__my_pipeline__"
                ),
            }
        ),
    ),
)


            # redution_tiled = structured.TileReductionUsingForOp(
            #     match(module_op, ["linalg.matmul"]),
            #     sizes=reduce_tile,
            #     # num_threads=reduce_tile
            # )
```


## warp and lane tile
```python
warp_mapping = Attribute.parse(
    "[ #gpu.warp<linear_dim_1>, #gpu.warp<linear_dim_0> ]"
)
thread_mapping = Attribute.parse(
    "[ #gpu.lane<linear_dim_1>, #gpu.lane<linear_dim_0> ]"
)
with print_op_context(module_op, "warp tile"):
    warp_tile_op = structured.TileUsingForallOp(
        match(module_op, ops=["linalg.matmul"]),  # tiled_op_type
        match(module_op, ops=["scf.forall"]),  # loops_type
        match(module_op, ops=["linalg.matmul"]),
        # num_threads=[2, 4, 4],
        tile_sizes=warp_tile,
        mapping=warp_mapping,
    )

with print_op_context(module_op, "tile level 1"):
    level_1_op = structured.TileUsingForallOp(
        match(module_op, ops=["linalg.matmul"]),  # tiled_op_type
        match(module_op, ops=["scf.forall"]),  # loops_type
        match(module_op, ops=["linalg.matmul"]),
        # num_threads=[2, 4, 4],
        tile_sizes=tiling_level_1,
        mapping=thread_mapping,
    )
```