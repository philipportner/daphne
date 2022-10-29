#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/Passes.h"
#include "compiler/CompilerUtils.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <utility>
#include <vector>
#include <iostream>


template <typename UnaryOp, typename IOp, typename FOp>
struct UnaryOpLowering : public mlir::OpConversionPattern<UnaryOp>
{

    mlir::LogicalResult
    matchAndRewrite(UnaryOp op, mlir::Value operand,
                    mlir::ConversionPatternRewriter &rewriter) const override
    {
        std::cout << "unary op\n";

        mlir::Type type = op.getType();
        if (type.isa<mlir::IntegerType>()) {
            rewriter.replaceOpWithNewOp<IOp>(op.getOperation(), operand);
        }
        else if (type.isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<FOp>(op.getOperation(), operand);
        }
        else {
            return mlir::failure();
        }
        return mlir::success();
    }
};
using AbsOpLowering = UnaryOpLowering<mlir::daphne::EwAbsOp, mlir::AbsFOp, mlir::AbsFOp>;

template <typename BinaryOp, typename ReplIOp, typename ReplFOp>
struct BinaryOpLowering : public mlir::OpConversionPattern<BinaryOp>
{
    using mlir::OpConversionPattern<BinaryOp>::OpConversionPattern;

    mlir::LogicalResult
    matchAndRewrite(BinaryOp op, mlir::ArrayRef<mlir::Value> operands,
                    mlir::ConversionPatternRewriter &rewriter) const override
    {
        std::cout << "op0 signless: " << operands[0].getType().isSignlessInteger() << "\n";
        std::cout << "op0 signed: "   << operands[0].getType().isSignedInteger() << "\n";
        std::cout << "op1 signless: " << operands[1].getType().isSignlessInteger() << "\n";
        std::cout << "op1 signed: "   << operands[1].getType().isSignedInteger() << "\n";

        mlir::Type type = op.getType();
        if (type.isa<mlir::IntegerType>()) {
            rewriter.replaceOpWithNewOp<ReplIOp>(op.getOperation(), operands);
        }
        else if (type.isa<mlir::FloatType>()) {
            rewriter.replaceOpWithNewOp<ReplFOp>(op.getOperation(), operands);
        }
        else {
            return mlir::failure();
        }
        return mlir::success();
    }
};
using AddOpLowering = BinaryOpLowering<mlir::daphne::EwAddOp, mlir::AddIOp, mlir::AddFOp>;
using SubOpLowering = BinaryOpLowering<mlir::daphne::EwSubOp, mlir::SubIOp, mlir::SubFOp>;
using MulOpLowering = BinaryOpLowering<mlir::daphne::EwMulOp, mlir::MulIOp, mlir::MulFOp>;

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
    // mlir::LowerToLLVMOptions llvmOptions(&getContext());
    // mlir::LLVMTypeConverter typeConverter(&getContext(), llvmOptions);
    //
    // typeConverter.addConversion([&](mlir::daphne::ConstantOp t)
    // {
    //     std::cout << "casting constant op\n";
    //     return mlir::IntegerType::get(t.getContext(), 64);
    // });



    target.addLegalOp<mlir::LLVM::AddOp, mlir::AddIOp, mlir::AddFOp>();
    target.addIllegalOp<mlir::daphne::EwAddOp, mlir::daphne::EwSubOp, mlir::daphne::EwMulOp>();

    // patterns.add<AddOpLowering>(&getContext());
    patterns.add<AddOpLowering, SubOpLowering, MulOpLowering>(&getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::daphne::createLowerScalarOpsPass(const DaphneUserConfig& cfg)
{
    return std::make_unique<LowerScalarOpsPass>(cfg);
}
