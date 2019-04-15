// 2019/04/15 - created by Tsung-Wei Huang

#pragma once

#define TF_STRINGIFY_HELPER(x) #x
#define TF_STRINGIFY(x) TF_STRINGIFY_HELPER(x)

// usage:
// #pragma message("content of macro: " TF_STRINGIFY(macro))
