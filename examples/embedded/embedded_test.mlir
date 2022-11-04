#map = affine_map<(d0) -> (0)>
#map1 = affine_map<(d0) -> ()>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<() -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {torch.debug_module_name = "train"} {
  func.func @forward(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<1x3xf32>, %arg3: tensor<1xf32>) -> (tensor<3xf32>, tensor<f32>, tensor<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-02 : f64
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %0 = tensor.empty() : tensor<1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.matvec ins(%arg2, %arg0 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1xf32>) -> tensor<1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel"]} ins(%2, %arg1 : tensor<1xf32>, tensor<f32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.addf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map2], iterator_types = ["parallel"]} ins(%3, %arg3 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.subf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %5 = tensor.empty() : tensor<i64>
    %6 = linalg.fill ins(%c1_i64 : i64) outs(%5 : tensor<i64>) -> tensor<i64>
    %7 = tensor.empty() : tensor<f32>
    %8 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%6 : tensor<i64>) outs(%7 : tensor<f32>) {
    ^bb0(%in: i64, %out: f32):
      %32 = arith.sitofp %in : i64 to f32
      linalg.yield %32 : f32
    } -> tensor<f32>
    %9 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel"]} ins(%8 : tensor<f32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel"]} ins(%9 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = math.powf %in, %cst_2 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel"]} ins(%11 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.mulf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map2], iterator_types = ["parallel"]} ins(%10, %12 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.mulf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %14 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
    %15 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["reduction"]} ins(%13 : tensor<1xf32>) outs(%14 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.addf %in, %out : f32
      linalg.yield %32 : f32
    } -> tensor<f32>
    %16 = tensor.empty() : tensor<3x1xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x3xf32>) outs(%16 : tensor<3x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1xf32>
    %18 = tensor.empty() : tensor<3xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<3xf32>) -> tensor<3xf32>
    %20 = linalg.matvec ins(%17, %13 : tensor<3x1xf32>, tensor<1xf32>) outs(%19 : tensor<3xf32>) -> tensor<3xf32>
    %21 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %22 = linalg.matvec ins(%arg2, %arg0 : tensor<1x3xf32>, tensor<3xf32>) outs(%21 : tensor<1xf32>) -> tensor<1xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel"]} ins(%22, %arg1 : tensor<1xf32>, tensor<f32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.addf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map, #map2], iterator_types = ["parallel"]} ins(%23, %arg3 : tensor<1xf32>, tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.subf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel"]} ins(%24 : tensor<1xf32>) outs(%0 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = math.powf %in, %cst_1 : f32
      linalg.yield %32 : f32
    } -> tensor<1xf32>
    %26 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
    %27 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["reduction"]} ins(%25 : tensor<1xf32>) outs(%26 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.addf %in, %out : f32
      linalg.yield %32 : f32
    } -> tensor<f32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%20 : tensor<3xf32>) outs(%18 : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.truncf %cst_0 : f64 to f32
      %33 = arith.mulf %in, %32 : f32
      linalg.yield %33 : f32
    } -> tensor<3xf32>
    %29 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg0, %28 : tensor<3xf32>, tensor<3xf32>) outs(%18 : tensor<3xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.subf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<3xf32>
    %30 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = []} ins(%15 : tensor<f32>) outs(%7 : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %32 = arith.truncf %cst_0 : f64 to f32
      %33 = arith.mulf %in, %32 : f32
      linalg.yield %33 : f32
    } -> tensor<f32>
    %31 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = []} ins(%arg1, %30 : tensor<f32>, tensor<f32>) outs(%7 : tensor<f32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %32 = arith.subf %in, %in_3 : f32
      linalg.yield %32 : f32
    } -> tensor<f32>
    return %29, %31, %27 : tensor<3xf32>, tensor<f32>, tensor<f32>
  }
}

