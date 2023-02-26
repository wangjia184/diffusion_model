"use strict";
/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.esmImportProvider = exports.getModuleProvider = void 0;
var fs = require("fs");
var path = require("path");
var custom_module_1 = require("./custom_module");
var model_parser_1 = require("./model_parser");
var util_1 = require("./util");
function getModuleProvider(opts) {
    return new ESMModuleProvider();
}
exports.getModuleProvider = getModuleProvider;
var ESMModuleProvider = /** @class */ (function () {
    function ESMModuleProvider() {
    }
    /**
     * Writes out custom tfjs module(s) to disk.
     */
    ESMModuleProvider.prototype.produceCustomTFJSModule = function (config) {
        var normalizedOutputPath = config.normalizedOutputPath;
        var moduleStrs = (0, custom_module_1.getCustomModuleString)(config, exports.esmImportProvider);
        fs.mkdirSync(normalizedOutputPath, { recursive: true });
        console.log("Will write custom tfjs modules to ".concat(normalizedOutputPath));
        var customTfjsFileName = 'custom_tfjs.js';
        var customTfjsCoreFileName = 'custom_tfjs_core.js';
        // Write a custom module for @tensorflow/tfjs and @tensorflow/tfjs-core
        fs.writeFileSync(path.join(normalizedOutputPath, customTfjsCoreFileName), moduleStrs.core);
        fs.writeFileSync(path.join(normalizedOutputPath, customTfjsFileName), moduleStrs.tfjs);
        // Write a custom module tfjs-core ops used by converter executors
        var kernelToOps;
        var mappingPath;
        try {
            mappingPath =
                require.resolve('@tensorflow/tfjs-converter/metadata/kernel2op.json');
            kernelToOps = JSON.parse(fs.readFileSync(mappingPath, 'utf-8'));
        }
        catch (e) {
            (0, util_1.bail)("Error loading kernel to ops mapping file ".concat(mappingPath));
        }
        var converterOps = (0, model_parser_1.getOpsForConfig)(config, kernelToOps);
        if (converterOps.length > 0) {
            var converterOpsModule = (0, custom_module_1.getCustomConverterOpsModule)(converterOps, exports.esmImportProvider);
            var customConverterOpsFileName = 'custom_ops_for_converter.js';
            fs.writeFileSync(path.join(normalizedOutputPath, customConverterOpsFileName), converterOpsModule);
        }
    };
    return ESMModuleProvider;
}());
/**
 * An import provider to generate custom esm modules.
 */
// Exported for tests.
exports.esmImportProvider = {
    importCoreStr: function (forwardModeOnly) {
        var importLines = [
            "import {registerKernel} from '@tensorflow/tfjs-core/dist/base';",
            "import '@tensorflow/tfjs-core/dist/base_side_effects';",
            "export * from '@tensorflow/tfjs-core/dist/base';"
        ];
        if (!forwardModeOnly) {
            importLines.push("import {registerGradient} from '@tensorflow/tfjs-core/dist/base';");
        }
        return importLines.join('\n');
    },
    importConverterStr: function () {
        return "export * from '@tensorflow/tfjs-converter';";
    },
    importBackendStr: function (backend) {
        var backendPkg = getBackendPath(backend);
        return "export * from '".concat(backendPkg, "/dist/base';");
    },
    importKernelStr: function (kernelName, backend) {
        var backendPkg = getBackendPath(backend);
        var kernelConfigId = "".concat(kernelName, "_").concat(backend);
        var importPath = "".concat(backendPkg, "/dist/kernels/").concat(kernelName);
        var importStatement = "import {".concat((0, util_1.kernelNameToVariableName)(kernelName), "Config as ").concat(kernelConfigId, "} from '").concat(importPath, "';");
        return { importPath: importPath, importStatement: importStatement, kernelConfigId: kernelConfigId };
    },
    importGradientConfigStr: function (kernelName) {
        var gradConfigId = "".concat((0, util_1.kernelNameToVariableName)(kernelName), "GradConfig");
        var importPath = "@tensorflow/tfjs-core/dist/gradients/".concat(kernelName, "_grad");
        var importStatement = "import {".concat(gradConfigId, "} from '").concat(importPath, "';");
        return { importPath: importPath, importStatement: importStatement, gradConfigId: gradConfigId };
    },
    importOpForConverterStr: function (opSymbol) {
        var opFileName = (0, util_1.opNameToFileName)(opSymbol);
        return "export {".concat(opSymbol, "} from '@tensorflow/tfjs-core/dist/ops/").concat(opFileName, "';");
    },
    importNamespacedOpsForConverterStr: function (namespace, opSymbols) {
        var result = [];
        for (var _i = 0, opSymbols_1 = opSymbols; _i < opSymbols_1.length; _i++) {
            var opSymbol = opSymbols_1[_i];
            var opFileName = (0, util_1.opNameToFileName)(opSymbol);
            var opAlias = "".concat(opSymbol, "_").concat(namespace);
            result.push("import {".concat(opSymbol, " as ").concat(opAlias, "} from '@tensorflow/tfjs-core/dist/ops/").concat(namespace, "/").concat(opFileName, "';"));
        }
        result.push("export const ".concat(namespace, " = {"));
        for (var _a = 0, opSymbols_2 = opSymbols; _a < opSymbols_2.length; _a++) {
            var opSymbol = opSymbols_2[_a];
            var opAlias = "".concat(opSymbol, "_").concat(namespace);
            result.push("\t".concat(opSymbol, ": ").concat(opAlias, ","));
        }
        result.push("};");
        return result.join('\n');
    },
    validateImportPath: function (importPath) {
        try {
            require.resolve(importPath);
            return true;
        }
        catch (e) {
            return false;
        }
    }
};
function getBackendPath(backend) {
    switch (backend) {
        case 'cpu':
            return '@tensorflow/tfjs-backend-cpu';
        case 'webgl':
            return '@tensorflow/tfjs-backend-webgl';
        case 'wasm':
            return '@tensorflow/tfjs-backend-wasm';
        default:
            throw new Error("Unsupported backend ".concat(backend));
    }
}
//# sourceMappingURL=esm_module_provider.js.map