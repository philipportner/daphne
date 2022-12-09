#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "compiler/CompilerUtils.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>


template <typename BinaryOp, typename IOp, typename FOp>
struct ScalarOpLowering : public mlir::OpConversionPattern<BinaryOp> {
    using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(
        BinaryOp op, mlir::ArrayRef<mlir::Value> operands,
        mlir::ConversionPatternRewriter &rewriter) const override {

        mlir::Type type = op.getType();
        type.dump();
        std::cout <<"\n";

        if (type.isa<mlir::IntegerType>()) {
            rewriter.replaceOpWithNewOp<IOp>(op.getOperation(), operands);
        } else if (type.isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(), operands);
        } else {
            return mlir::failure();
        }
        return mlir::success();
    }
};
using AddOpLowering = ScalarOpLowering<mlir::daphne::EwAddOp, mlir::AddIOp, mlir::AddFOp>;
using SubOpLowering = ScalarOpLowering<mlir::daphne::EwSubOp, mlir::SubIOp, mlir::SubFOp>;
using MulOpLowering = ScalarOpLowering<mlir::daphne::EwMulOp, mlir::MulIOp, mlir::MulFOp>;
using DivOpLowering = ScalarOpLowering<mlir::daphne::EwDivOp, mlir::DivFOp, mlir::DivFOp>;
// FIXME: IPowIOp has been added to MathOps.td with 08b4cf3 Aug 10
using PowOpLowering = ScalarOpLowering<mlir::daphne::EwPowOp, mlir::math::PowFOp, mlir::math::PowFOp>;
// FIXME: AbsIOp  has been added to MathOps.td with 7d9fc95 Aug 08
using AbsOpLowering = ScalarOpLowering<mlir::daphne::EwAbsOp, mlir::AbsFOp, mlir::AbsFOp>;
// FIXME
using LnOpLowering = ScalarOpLowering<mlir::daphne::EwLnOp, mlir::math::LogOp, mlir::math::LogOp>;

namespace
{
    struct LowerScalarOpsPass
    : public mlir::PassWrapper<LowerScalarOpsPass, mlir::OperationPass<mlir::ModuleOp>>
    {
		explicit LowerScalarOpsPass(const DaphneUserConfig& cfg) : cfg(cfg) { }
		const DaphneUserConfig& cfg;

        void getDependentDialects(mlir::DialectRegistry & registry) const override
        {
            registry.insert<mlir::LLVM::LLVMDialect/*, scf::SCFDialect*/>();
        }
        void runOnOperation() final;
    };
} // end anonymous namespace

void LowerScalarOpsPass::runOnOperation()
{
    auto module = getOperation();
    mlir::OwningRewritePatternList patterns(&getContext());

    mlir::LLVMConversionTarget target(getContext());
    mlir::LowerToLLVMOptions llvmOptions(&getContext());
    mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
    mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

    target.addLegalOp<mlir::LLVM::AddOp, mlir::AddIOp, mlir::AddFOp, mlir::FuncOp>();
    // target.addIllegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp, mlir::daphne::EwMulOp>();

    patterns.add<AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                 PowOpLowering, AbsOpLowering>(typeConverter, &getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerScalarOpsPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<LowerScalarOpsPass>(cfg);
}
