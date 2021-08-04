/*
 * Copyright 2021 The DAPHNE Consortium
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

#include <ir/daphneir/Daphne.h>

#include <string>
#include <vector>
#include <stdexcept>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferTypesOpInterface.cpp.inc>
}

using namespace mlir;

// ****************************************************************************
// General utility functions
// ****************************************************************************

Type getFrameColumnTypeByLabel(daphne::FrameType ft, Value label) {
    // TODO Use getConstantString from DaphneInferFrameLabelsOpInterface.cpp.
    if(auto co = llvm::dyn_cast<daphne::ConstantOp>(label.getDefiningOp())) {
        if(auto strAttr = co.value().dyn_cast<StringAttr>()) {
            std::string label = strAttr.getValue().str();
            std::vector<std::string> * labels = ft.getLabels();
            if(labels) {
                // The column labels are known, so we search for the specified
                // label.
                std::vector<Type> colTypes = ft.getColumnTypes();
                for(size_t i = 0; i < colTypes.size(); i++)
                    if((*labels)[i] == label)
                        // Found the label.
                        return colTypes[i];
                // Did not find the label.
                throw std::runtime_error("the specified label was not found");
            }
            else
                // The column labels are unknown, so we cannot tell what type
                // the column with the specified label has.
                return daphne::UnknownType::get(ft.getContext());
        }
    }
    throw std::runtime_error(
            "the specified label must be a constant of string type"
    );
}

// ****************************************************************************
// Type inference utility functions
// ****************************************************************************
// For families of operations.

template<class EwCmpOp>
void inferTypes_EwCmpOp(EwCmpOp * op) {
    Type lhsType = op->lhs().getType();
    Type rhsType = op->rhs().getType();
    Type t;
    if(lhsType.isa<daphne::MatrixType>())
        t = lhsType;
    else if(rhsType.isa<daphne::MatrixType>())
        t = rhsType;
    else {
        Builder builder(op->getContext());
        t = builder.getI1Type();
    }
    op->getResult().setType(t);
}

// ****************************************************************************
// Type inference implementations
// ****************************************************************************

void daphne::ExtractColOp::inferTypes() {
    if(auto ft = source().getType().dyn_cast<daphne::FrameType>()) {
        Type vt = getFrameColumnTypeByLabel(ft, selectedCols());
        getResult().setType(daphne::MatrixType::get(getContext(), vt));
    }
    else
        throw std::runtime_error(
                "currently, ExtractColOp can only infer its type for frame "
                "inputs"
        );
}

void daphne::CreateFrameOp::inferTypes() {
    std::vector<Type> colTypes;
    for(Value col : cols())
        colTypes.push_back(col.getType().dyn_cast<daphne::MatrixType>().getElementType());
    getResult().setType(daphne::FrameType::get(getContext(), colTypes));
}

void daphne::EwEqOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwNeqOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwLtOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwLeOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwGtOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::EwGeOp::inferTypes() {
    return inferTypes_EwCmpOp(this);
}

void daphne::FilterRowOp::inferTypes() {
    getResult().setType(source().getType());
}

void daphne::GroupJoinOp::inferTypes() {
    daphne::FrameType lhsFt = lhs().getType().dyn_cast<daphne::FrameType>();
    daphne::FrameType rhsFt = rhs().getType().dyn_cast<daphne::FrameType>();
    Type lhsOnType = getFrameColumnTypeByLabel(lhsFt, lhsOn());
    Type rhsAggType = getFrameColumnTypeByLabel(rhsFt, rhsAgg());
    
    MLIRContext * ctx = getContext();
    Builder builder(ctx);
    getResult(0).setType(daphne::FrameType::get(ctx, {lhsOnType, rhsAggType}));
    getResult(1).setType(daphne::MatrixType::get(ctx, builder.getIndexType()));
}