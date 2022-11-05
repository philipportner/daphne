/*
 * Copyright 2022 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <api/cli/Utils.h>
#include <tags.h>

#include <catch.hpp>
#include <sstream>
#include <string>

#include "api/cli/StatusCode.h"

const std::string dirPath = "test/runtime/local/lowering/";

TEST_CASE("ewBinaryAddScalar", TAG_KERNELS) {
    std::stringstream out;
    std::stringstream err;

    // `daphne --explain llvm $scriptFilePath`
    int status = runDaphne(out, err, "--explain", "llvm", (dirPath + "add.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    // --lowering-scalar not passed
    // make sure EwAddOp is correctly lowered to kernel call
    // PrintIRPass outputs to stderr
    REQUIRE_THAT(err.str(), Catch::Contains("llvm.call @_ewAdd__"));
    REQUIRE_THAT(err.str(), !Catch::Contains("llvm.add"));
    CHECK(out.str() == "3\n");

    out.str(std::string());
    err.str(std::string());

    // `daphne --explain llvm --scalar-lowering $scriptFilePath`
    status = runDaphne(out, err, "--explain", "llvm", "--scalar-lowering", (dirPath + "add.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    // --lowering-scalar
    // make sure EwAddOp is no longer lowered to kernel call
    REQUIRE_THAT(err.str(), !Catch::Contains("llvm.call @_ewAdd__"));
    REQUIRE_THAT(err.str(), Catch::Contains("llvm.add"));
    CHECK(out.str() == "3\n");
}

TEST_CASE("ewBinarySubScalar", TAG_KERNELS) {
    std::stringstream out;
    std::stringstream err;

    // `daphne --explain llvm $scriptFilePath`
    int status = runDaphne(out, err, "--explain", "llvm", (dirPath + "sub.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    // --lowering-scalar not passed
    // make sure EwSubOp is correctly lowered to kernel call
    // PrintIRPass outputs to stderr
    REQUIRE_THAT(err.str(), Catch::Contains("llvm.call @_ewSub__"));
    REQUIRE_THAT(err.str(), !Catch::Contains("llvm.sub"));
    CHECK(out.str() == "-1\n");

    out.str(std::string());
    err.str(std::string());

    // `daphne --explain llvm --scalar-lowering $scriptFilePath`
    status = runDaphne(out, err, "--explain", "llvm", "--scalar-lowering", (dirPath + "sub.daphne").c_str());
    CHECK(status == StatusCode::SUCCESS);

    // --lowering-scalar
    // make sure EwSubOp is no longer lowered to kernel call
    REQUIRE_THAT(err.str(), !Catch::Contains("llvm.call @_ewSub__"));
    REQUIRE_THAT(err.str(), Catch::Contains("llvm.sub"));
    CHECK(out.str() == "-1\n");
}
