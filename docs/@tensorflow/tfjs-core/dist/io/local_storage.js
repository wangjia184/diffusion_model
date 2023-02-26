/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import '../flags';
import { env } from '../environment';
import { assert } from '../util';
import { arrayBufferToBase64String, base64StringToArrayBuffer, getModelArtifactsInfoForJSON } from './io_utils';
import { IORouterRegistry } from './router_registry';
const PATH_SEPARATOR = '/';
const PATH_PREFIX = 'tensorflowjs_models';
const INFO_SUFFIX = 'info';
const MODEL_TOPOLOGY_SUFFIX = 'model_topology';
const WEIGHT_SPECS_SUFFIX = 'weight_specs';
const WEIGHT_DATA_SUFFIX = 'weight_data';
const MODEL_METADATA_SUFFIX = 'model_metadata';
/**
 * Purge all tensorflow.js-saved model artifacts from local storage.
 *
 * @returns Paths of the models purged.
 */
export function purgeLocalStorageArtifacts() {
    if (!env().getBool('IS_BROWSER') || typeof window === 'undefined' ||
        typeof window.localStorage === 'undefined') {
        throw new Error('purgeLocalStorageModels() cannot proceed because local storage is ' +
            'unavailable in the current environment.');
    }
    const LS = window.localStorage;
    const purgedModelPaths = [];
    for (let i = 0; i < LS.length; ++i) {
        const key = LS.key(i);
        const prefix = PATH_PREFIX + PATH_SEPARATOR;
        if (key.startsWith(prefix) && key.length > prefix.length) {
            LS.removeItem(key);
            const modelName = getModelPathFromKey(key);
            if (purgedModelPaths.indexOf(modelName) === -1) {
                purgedModelPaths.push(modelName);
            }
        }
    }
    return purgedModelPaths;
}
function getModelKeys(path) {
    return {
        info: [PATH_PREFIX, path, INFO_SUFFIX].join(PATH_SEPARATOR),
        topology: [PATH_PREFIX, path, MODEL_TOPOLOGY_SUFFIX].join(PATH_SEPARATOR),
        weightSpecs: [PATH_PREFIX, path, WEIGHT_SPECS_SUFFIX].join(PATH_SEPARATOR),
        weightData: [PATH_PREFIX, path, WEIGHT_DATA_SUFFIX].join(PATH_SEPARATOR),
        modelMetadata: [PATH_PREFIX, path, MODEL_METADATA_SUFFIX].join(PATH_SEPARATOR)
    };
}
function removeItems(keys) {
    for (const key of Object.values(keys)) {
        window.localStorage.removeItem(key);
    }
}
/**
 * Get model path from a local-storage key.
 *
 * E.g., 'tensorflowjs_models/my/model/1/info' --> 'my/model/1'
 *
 * @param key
 */
function getModelPathFromKey(key) {
    const items = key.split(PATH_SEPARATOR);
    if (items.length < 3) {
        throw new Error(`Invalid key format: ${key}`);
    }
    return items.slice(1, items.length - 1).join(PATH_SEPARATOR);
}
function maybeStripScheme(key) {
    return key.startsWith(BrowserLocalStorage.URL_SCHEME) ?
        key.slice(BrowserLocalStorage.URL_SCHEME.length) :
        key;
}
/**
 * IOHandler subclass: Browser Local Storage.
 *
 * See the doc string to `browserLocalStorage` for more details.
 */
export class BrowserLocalStorage {
    constructor(modelPath) {
        if (!env().getBool('IS_BROWSER') || typeof window === 'undefined' ||
            typeof window.localStorage === 'undefined') {
            // TODO(cais): Add more info about what IOHandler subtypes are
            // available.
            //   Maybe point to a doc page on the web and/or automatically determine
            //   the available IOHandlers and print them in the error message.
            throw new Error('The current environment does not support local storage.');
        }
        this.LS = window.localStorage;
        if (modelPath == null || !modelPath) {
            throw new Error('For local storage, modelPath must not be null, undefined or empty.');
        }
        this.modelPath = modelPath;
        this.keys = getModelKeys(this.modelPath);
    }
    /**
     * Save model artifacts to browser local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @param modelArtifacts The model artifacts to be stored.
     * @returns An instance of SaveResult.
     */
    async save(modelArtifacts) {
        if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
            throw new Error('BrowserLocalStorage.save() does not support saving model topology ' +
                'in binary formats yet.');
        }
        else {
            const topology = JSON.stringify(modelArtifacts.modelTopology);
            const weightSpecs = JSON.stringify(modelArtifacts.weightSpecs);
            const modelArtifactsInfo = getModelArtifactsInfoForJSON(modelArtifacts);
            try {
                this.LS.setItem(this.keys.info, JSON.stringify(modelArtifactsInfo));
                this.LS.setItem(this.keys.topology, topology);
                this.LS.setItem(this.keys.weightSpecs, weightSpecs);
                this.LS.setItem(this.keys.weightData, arrayBufferToBase64String(modelArtifacts.weightData));
                // Note that JSON.stringify doesn't write out keys that have undefined
                // values, so for some keys, we set undefined instead of a null-ish
                // value.
                const metadata = {
                    format: modelArtifacts.format,
                    generatedBy: modelArtifacts.generatedBy,
                    convertedBy: modelArtifacts.convertedBy,
                    signature: modelArtifacts.signature != null ?
                        modelArtifacts.signature :
                        undefined,
                    userDefinedMetadata: modelArtifacts.userDefinedMetadata != null ?
                        modelArtifacts.userDefinedMetadata :
                        undefined,
                    modelInitializer: modelArtifacts.modelInitializer != null ?
                        modelArtifacts.modelInitializer :
                        undefined,
                    initializerSignature: modelArtifacts.initializerSignature != null ?
                        modelArtifacts.initializerSignature :
                        undefined,
                    trainingConfig: modelArtifacts.trainingConfig != null ?
                        modelArtifacts.trainingConfig :
                        undefined
                };
                this.LS.setItem(this.keys.modelMetadata, JSON.stringify(metadata));
                return { modelArtifactsInfo };
            }
            catch (err) {
                // If saving failed, clean up all items saved so far.
                removeItems(this.keys);
                throw new Error(`Failed to save model '${this.modelPath}' to local storage: ` +
                    `size quota being exceeded is a possible cause of this failure: ` +
                    `modelTopologyBytes=${modelArtifactsInfo.modelTopologyBytes}, ` +
                    `weightSpecsBytes=${modelArtifactsInfo.weightSpecsBytes}, ` +
                    `weightDataBytes=${modelArtifactsInfo.weightDataBytes}.`);
            }
        }
    }
    /**
     * Load a model from local storage.
     *
     * See the documentation to `browserLocalStorage` for details on the saved
     * artifacts.
     *
     * @returns The loaded model (if loading succeeds).
     */
    async load() {
        const info = JSON.parse(this.LS.getItem(this.keys.info));
        if (info == null) {
            throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);
        }
        if (info.modelTopologyType !== 'JSON') {
            throw new Error('BrowserLocalStorage does not support loading non-JSON model ' +
                'topology yet.');
        }
        const out = {};
        // Load topology.
        const topology = JSON.parse(this.LS.getItem(this.keys.topology));
        if (topology == null) {
            throw new Error(`In local storage, the topology of model '${this.modelPath}' ` +
                `is missing.`);
        }
        out.modelTopology = topology;
        // Load weight specs.
        const weightSpecs = JSON.parse(this.LS.getItem(this.keys.weightSpecs));
        if (weightSpecs == null) {
            throw new Error(`In local storage, the weight specs of model '${this.modelPath}' ` +
                `are missing.`);
        }
        out.weightSpecs = weightSpecs;
        // Load meta-data fields.
        const metadataString = this.LS.getItem(this.keys.modelMetadata);
        if (metadataString != null) {
            const metadata = JSON.parse(metadataString);
            out.format = metadata.format;
            out.generatedBy = metadata.generatedBy;
            out.convertedBy = metadata.convertedBy;
            if (metadata.signature != null) {
                out.signature = metadata.signature;
            }
            if (metadata.userDefinedMetadata != null) {
                out.userDefinedMetadata = metadata.userDefinedMetadata;
            }
            if (metadata.modelInitializer != null) {
                out.modelInitializer = metadata.modelInitializer;
            }
            if (metadata.initializerSignature != null) {
                out.initializerSignature = metadata.initializerSignature;
            }
            if (metadata.trainingConfig != null) {
                out.trainingConfig = metadata.trainingConfig;
            }
        }
        // Load weight data.
        const weightDataBase64 = this.LS.getItem(this.keys.weightData);
        if (weightDataBase64 == null) {
            throw new Error(`In local storage, the binary weight values of model ` +
                `'${this.modelPath}' are missing.`);
        }
        out.weightData = base64StringToArrayBuffer(weightDataBase64);
        return out;
    }
}
BrowserLocalStorage.URL_SCHEME = 'localstorage://';
export const localStorageRouter = (url) => {
    if (!env().getBool('IS_BROWSER')) {
        return null;
    }
    else {
        if (!Array.isArray(url) && url.startsWith(BrowserLocalStorage.URL_SCHEME)) {
            return browserLocalStorage(url.slice(BrowserLocalStorage.URL_SCHEME.length));
        }
        else {
            return null;
        }
    }
};
IORouterRegistry.registerSaveRouter(localStorageRouter);
IORouterRegistry.registerLoadRouter(localStorageRouter);
/**
 * Factory function for local storage IOHandler.
 *
 * This `IOHandler` supports both `save` and `load`.
 *
 * For each model's saved artifacts, four items are saved to local storage.
 *   - `${PATH_SEPARATOR}/${modelPath}/info`: Contains meta-info about the
 *     model, such as date saved, type of the topology, size in bytes, etc.
 *   - `${PATH_SEPARATOR}/${modelPath}/topology`: Model topology. For Keras-
 *     style models, this is a stringized JSON.
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_specs`: Weight specs of the
 *     model, can be used to decode the saved binary weight values (see
 *     item below).
 *   - `${PATH_SEPARATOR}/${modelPath}/weight_data`: Concatenated binary
 *     weight values, stored as a base64-encoded string.
 *
 * Saving may throw an `Error` if the total size of the artifacts exceed the
 * browser-specific quota.
 *
 * @param modelPath A unique identifier for the model to be saved. Must be a
 *   non-empty string.
 * @returns An instance of `IOHandler`, which can be used with, e.g.,
 *   `tf.Model.save`.
 */
export function browserLocalStorage(modelPath) {
    return new BrowserLocalStorage(modelPath);
}
export class BrowserLocalStorageManager {
    constructor() {
        assert(env().getBool('IS_BROWSER'), () => 'Current environment is not a web browser');
        assert(typeof window === 'undefined' ||
            typeof window.localStorage !== 'undefined', () => 'Current browser does not appear to support localStorage');
        this.LS = window.localStorage;
    }
    async listModels() {
        const out = {};
        const prefix = PATH_PREFIX + PATH_SEPARATOR;
        const suffix = PATH_SEPARATOR + INFO_SUFFIX;
        for (let i = 0; i < this.LS.length; ++i) {
            const key = this.LS.key(i);
            if (key.startsWith(prefix) && key.endsWith(suffix)) {
                const modelPath = getModelPathFromKey(key);
                out[modelPath] = JSON.parse(this.LS.getItem(key));
            }
        }
        return out;
    }
    async removeModel(path) {
        path = maybeStripScheme(path);
        const keys = getModelKeys(path);
        if (this.LS.getItem(keys.info) == null) {
            throw new Error(`Cannot find model at path '${path}'`);
        }
        const info = JSON.parse(this.LS.getItem(keys.info));
        removeItems(keys);
        return info;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9jYWxfc3RvcmFnZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vbG9jYWxfc3RvcmFnZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLFVBQVUsQ0FBQztBQUNsQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFFbkMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUMvQixPQUFPLEVBQUMseUJBQXlCLEVBQUUseUJBQXlCLEVBQUUsNEJBQTRCLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDOUcsT0FBTyxFQUFXLGdCQUFnQixFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFHN0QsTUFBTSxjQUFjLEdBQUcsR0FBRyxDQUFDO0FBQzNCLE1BQU0sV0FBVyxHQUFHLHFCQUFxQixDQUFDO0FBQzFDLE1BQU0sV0FBVyxHQUFHLE1BQU0sQ0FBQztBQUMzQixNQUFNLHFCQUFxQixHQUFHLGdCQUFnQixDQUFDO0FBQy9DLE1BQU0sbUJBQW1CLEdBQUcsY0FBYyxDQUFDO0FBQzNDLE1BQU0sa0JBQWtCLEdBQUcsYUFBYSxDQUFDO0FBQ3pDLE1BQU0scUJBQXFCLEdBQUcsZ0JBQWdCLENBQUM7QUFFL0M7Ozs7R0FJRztBQUNILE1BQU0sVUFBVSwwQkFBMEI7SUFDeEMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyxZQUFZLENBQUMsSUFBSSxPQUFPLE1BQU0sS0FBSyxXQUFXO1FBQzdELE9BQU8sTUFBTSxDQUFDLFlBQVksS0FBSyxXQUFXLEVBQUU7UUFDOUMsTUFBTSxJQUFJLEtBQUssQ0FDWCxvRUFBb0U7WUFDcEUseUNBQXlDLENBQUMsQ0FBQztLQUNoRDtJQUNELE1BQU0sRUFBRSxHQUFHLE1BQU0sQ0FBQyxZQUFZLENBQUM7SUFDL0IsTUFBTSxnQkFBZ0IsR0FBYSxFQUFFLENB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QUFDLFVBQVUsQ0FBQyxNQUFNLENBQUMsSUFBSSxHQUFHLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxFQUFFO2dCQUNsRCxNQUFNLFNBQVMsR0FBRyxtQkFBbUIsQ0FBQyxHQUFHLENBQUMsQ0FBQztnQkFDM0MsR0FBRyxDQUFDLFNBQVMsQ0FBQyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQXVCLENBQUM7YUFDekU7U0FDRjtRQUNELE9BQU8sR0FBRyxDQUFDO0lBQ2IsQ0FBQztJQUVELEtBQUssQ0FBQyxXQUFXLENBQUMsSUFBWTtRQUM1QixJQUFJLEdBQUcsZ0JBQWdCLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDOUIsTUFBTSxJQUFJLEdBQUcsWUFBWSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2hDLElBQUksSUFBSSxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxJQUFJLElBQUksRUFBRTtZQUN0QyxNQUFNLElBQUksS0FBSyxDQUFDLDhCQUE4QixJQUFJLEdBQUcsQ0FBQyxDQUFDO1NBQ3hEO1FBQ0QsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQXVCLENBQUM7UUFDMUUsV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2xCLE9BQU8sSUFBSSxDQUFDO0lBQ2QsQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgJy4uL2ZsYWdzJztcbmltcG9ydCB7ZW52fSBmcm9tICcuLi9lbnZpcm9ubWVudCc7XG5cbmltcG9ydCB7YXNzZXJ0fSBmcm9tICcuLi91dGlsJztcbmltcG9ydCB7YXJyYXlCdWZmZXJUb0Jhc2U2NFN0cmluZywgYmFzZTY0U3RyaW5nVG9BcnJheUJ1ZmZlciwgZ2V0TW9kZWxBcnRpZmFjdHNJbmZvRm9ySlNPTn0gZnJvbSAnLi9pb191dGlscyc7XG5pbXBvcnQge0lPUm91dGVyLCBJT1JvdXRlclJlZ2lzdHJ5fSBmcm9tICcuL3JvdXRlcl9yZWdpc3RyeSc7XG5pbXBvcnQge0lPSGFuZGxlciwgTW9kZWxBcnRpZmFjdHMsIE1vZGVsQXJ0aWZhY3RzSW5mbywgTW9kZWxKU09OLCBNb2RlbFN0b3JlTWFuYWdlciwgU2F2ZVJlc3VsdH0gZnJvbSAnLi90eXBlcyc7XG5cbmNvbnN0IFBBVEhfU0VQQVJBVE9SID0gJy8nO1xuY29uc3QgUEFUSF9QUkVGSVggPSAndGVuc29yZmxvd2pzX21vZGVscyc7XG5jb25zdCBJTkZPX1NVRkZJWCA9ICdpbmZvJztcbmNvbnN0IE1PREVMX1RPUE9MT0dZX1NVRkZJWCA9ICdtb2RlbF90b3BvbG9neSc7XG5jb25zdCBXRUlHSFRfU1BFQ1NfU1VGRklYID0gJ3dlaWdodF9zcGVjcyc7XG5jb25zdCBXRUlHSFRfREFUQV9TVUZGSVggPSAnd2VpZ2h0X2RhdGEnO1xuY29uc3QgTU9ERUxfTUVUQURBVEFfU1VGRklYID0gJ21vZGVsX21ldGFkYXRhJztcblxuLyoqXG4gKiBQdXJnZSBhbGwgdGVuc29yZmxvdy5qcy1zYXZlZCBtb2RlbCBhcnRpZmFjdHMgZnJvbSBsb2NhbCBzdG9yYWdlLlxuICpcbiAqIEByZXR1cm5zIFBhdGhzIG9mIHRoZSBtb2RlbHMgcHVyZ2VkLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcHVyZ2VMb2NhbFN0b3JhZ2VBcnRpZmFjdHMoKTogc3RyaW5nW10ge1xuICBpZiAoIWVudigpLmdldEJvb2woJ0lTX0JST1dTRVInKSB8fCB0eXBlb2Ygd2luZG93ID09PSAndW5kZWZpbmVkJyB8fFxuICAgICAgdHlwZW9mIHdpbmRvdy5sb2NhbFN0b3JhZ2UgPT09ICd1bmRlZmluZWQnKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAncHVyZ2VMb2NhbFN0b3JhZ2VNb2RlbHMoKSBjYW5ub3QgcHJvY2VlZCBiZWNhdXNlIGxvY2FsIHN0b3JhZ2UgaXMgJyArXG4gICAgICAgICd1bmF2YWlsYWJsZSBpbiB0aGUgY3VycmVudCBlbnZpcm9ubWVudC4nKTtcbiAgfVxuICBjb25zdCBMUyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2U7XG4gIGNvbnN0IHB1cmdlZE1vZGVsUGF0aHM6IHN0cmluZ1tdID0gW107XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgTFMubGVuZ3RoOyArK2kpIHtcbiAgICBjb25zdCBrZXkgPSBMUy5rZXkoaSk7XG4gICAgY29uc3QgcHJlZml4ID0gUEFUSF9QUkVGSVggKyBQQVRIX1NFUEFSQVRPUjtcbiAgICBpZiAoa2V5LnN0YXJ0c1dpdGgocHJlZml4KSAmJiBrZXkubGVuZ3RoID4gcHJlZml4Lmxlbmd0aCkge1xuICAgICAgTFMucmVtb3ZlSXRlbShrZXkpO1xuICAgICAgY29uc3QgbW9kZWxOYW1lID0gZ2V0TW9kZWxQYXRoRnJvbUtleShrZXkpO1xuICAgICAgaWYgKHB1cmdlZE1vZGVsUGF0aHMuaW5kZXhPZihtb2RlbE5hbWUpID09PSAtMSkge1xuICAgICAgICBwdXJnZWRNb2RlbFBhdGhzLnB1c2gobW9kZWxOYW1lKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cbiAgcmV0dXJuIHB1cmdlZE1vZGVsUGF0aHM7XG59XG5cbnR5cGUgTG9jYWxTdG9yYWdlS2V5cyA9IHtcbiAgLyoqIEtleSBvZiB0aGUgbG9jYWxTdG9yYWdlIGVudHJ5IHN0b3JpbmcgYE1vZGVsQXJ0aWZhY3RzSW5mb2AuICovXG4gIGluZm86IHN0cmluZyxcbiAgLyoqXG4gICAqIEtleSBvZiB0aGUgbG9jYWxTdG9yYWdlIGVudHJ5IHN0b3JpbmcgdGhlICdtb2RlbFRvcG9sb2d5JyBrZXkgb2ZcbiAgICogYG1vZGVsLmpzb25gXG4gICAqL1xuICB0b3BvbG9neTogc3RyaW5nLFxuICAvKipcbiAgICogS2V5IG9mIHRoZSBsb2NhbFN0b3JhZ2UgZW50cnkgc3RvcmluZyB0aGUgYHdlaWdodHNNYW5pZmVzdC53ZWlnaHRzYCBlbnRyaWVzXG4gICAqIG9mIGBtb2RlbC5qc29uYFxuICAgKi9cbiAgd2VpZ2h0U3BlY3M6IHN0cmluZyxcbiAgLyoqIEtleSBvZiB0aGUgbG9jYWxTdG9yYWdlIGVudHJ5IHN0b3JpbmcgdGhlIHdlaWdodCBkYXRhIGluIEJhc2U2NCAqL1xuICB3ZWlnaHREYXRhOiBzdHJpbmcsXG4gIC8qKlxuICAgKiBLZXkgb2YgdGhlIGxvY2FsU3RvcmFnZSBlbnRyeSBzdG9yaW5nIHRoZSByZW1haW5pbmcgZmllbGRzIG9mIGBtb2RlbC5qc29uYFxuICAgKiBAc2VlIHtAbGluayBNb2RlbE1ldGFkYXRhfVxuICAgKi9cbiAgbW9kZWxNZXRhZGF0YTogc3RyaW5nLFxufTtcblxudHlwZSBNb2RlbE1ldGFkYXRhID0gT21pdDxNb2RlbEpTT04sICdtb2RlbFRvcG9sb2d5J3wnd2VpZ2h0c01hbmlmZXN0Jz47XG5cbmZ1bmN0aW9uIGdldE1vZGVsS2V5cyhwYXRoOiBzdHJpbmcpOiBMb2NhbFN0b3JhZ2VLZXlzIHtcbiAgcmV0dXJuIHtcbiAgICBpbmZvOiBbUEFUSF9QUkVGSVgsIHBhdGgsIElORk9fU1VGRklYXS5qb2luKFBBVEhfU0VQQVJBVE9SKSxcbiAgICB0b3BvbG9neTogW1BBVEhfUFJFRklYLCBwYXRoLCBNT0RFTF9UT1BPTE9HWV9TVUZGSVhdLmpvaW4oUEFUSF9TRVBBUkFUT1IpLFxuICAgIHdlaWdodFNwZWNzOiBbUEFUSF9QUkVGSVgsIHBhdGgsIFdFSUdIVF9TUEVDU19TVUZGSVhdLmpvaW4oUEFUSF9TRVBBUkFUT1IpLFxuICAgIHdlaWdodERhdGE6IFtQQVRIX1BSRUZJWCwgcGF0aCwgV0VJR0hUX0RBVEFfU1VGRklYXS5qb2luKFBBVEhfU0VQQVJBVE9SKSxcbiAgICBtb2RlbE1ldGFkYXRhOlxuICAgICAgICBbUEFUSF9QUkVGSVgsIHBhdGgsIE1PREVMX01FVEFEQVRBX1NVRkZJWF0uam9pbihQQVRIX1NFUEFSQVRPUilcbiAgfTtcbn1cblxuZnVuY3Rpb24gcmVtb3ZlSXRlbXMoa2V5czogTG9jYWxTdG9yYWdlS2V5cyk6IHZvaWQge1xuICBmb3IgKGNvbnN0IGtleSBvZiBPYmplY3QudmFsdWVzKGtleXMpKSB7XG4gICAgd2luZG93LmxvY2FsU3RvcmFnZS5yZW1vdmVJdGVtKGtleSk7XG4gIH1cbn1cblxuLyoqXG4gKiBHZXQgbW9kZWwgcGF0aCBmcm9tIGEgbG9jYWwtc3RvcmFnZSBrZXkuXG4gKlxuICogRS5nLiwgJ3RlbnNvcmZsb3dqc19tb2RlbHMvbXkvbW9kZWwvMS9pbmZvJyAtLT4gJ215L21vZGVsLzEnXG4gKlxuICogQHBhcmFtIGtleVxuICovXG5mdW5jdGlvbiBnZXRNb2RlbFBhdGhGcm9tS2V5KGtleTogc3RyaW5nKSB7XG4gIGNvbnN0IGl0ZW1zID0ga2V5LnNwbGl0KFBBVEhfU0VQQVJBVE9SKTtcbiAgaWYgKGl0ZW1zLmxlbmd0aCA8IDMpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYEludmFsaWQga2V5IGZvcm1hdDogJHtrZXl9YCk7XG4gIH1cbiAgcmV0dXJuIGl0ZW1zLnNsaWNlKDEsIGl0ZW1zLmxlbmd0aCAtIDEpLmpvaW4oUEFUSF9TRVBBUkFUT1IpO1xufVxuXG5mdW5jdGlvbiBtYXliZVN0cmlwU2NoZW1lKGtleTogc3RyaW5nKSB7XG4gIHJldHVybiBrZXkuc3RhcnRzV2l0aChCcm93c2VyTG9jYWxTdG9yYWdlLlVSTF9TQ0hFTUUpID9cbiAgICAgIGtleS5zbGljZShCcm93c2VyTG9jYWxTdG9yYWdlLlVSTF9TQ0hFTUUubGVuZ3RoKSA6XG4gICAgICBrZXk7XG59XG5cbi8qKlxuICogSU9IYW5kbGVyIHN1YmNsYXNzOiBCcm93c2VyIExvY2FsIFN0b3JhZ2UuXG4gKlxuICogU2VlIHRoZSBkb2Mgc3RyaW5nIHRvIGBicm93c2VyTG9jYWxTdG9yYWdlYCBmb3IgbW9yZSBkZXRhaWxzLlxuICovXG5leHBvcnQgY2xhc3MgQnJvd3NlckxvY2FsU3RvcmFnZSBpbXBsZW1lbnRzIElPSGFuZGxlciB7XG4gIHByb3RlY3RlZCByZWFkb25seSBMUzogU3RvcmFnZTtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IG1vZGVsUGF0aDogc3RyaW5nO1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkga2V5czogTG9jYWxTdG9yYWdlS2V5cztcblxuICBzdGF0aWMgcmVhZG9ubHkgVVJMX1NDSEVNRSA9ICdsb2NhbHN0b3JhZ2U6Ly8nO1xuXG4gIGNvbnN0cnVjdG9yKG1vZGVsUGF0aDogc3RyaW5nKSB7XG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdJU19CUk9XU0VSJykgfHwgdHlwZW9mIHdpbmRvdyA9PT0gJ3VuZGVmaW5lZCcgfHxcbiAgICAgICAgdHlwZW9mIHdpbmRvdy5sb2NhbFN0b3JhZ2UgPT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAvLyBUT0RPKGNhaXMpOiBBZGQgbW9yZSBpbmZvIGFib3V0IHdoYXQgSU9IYW5kbGVyIHN1YnR5cGVzIGFyZVxuICAgICAgLy8gYXZhaWxhYmxlLlxuICAgICAgLy8gICBNYXliZSBwb2ludCB0byBhIGRvYyBwYWdlIG9uIHRoZSB3ZWIgYW5kL29yIGF1dG9tYXRpY2FsbHkgZGV0ZXJtaW5lXG4gICAgICAvLyAgIHRoZSBhdmFpbGFibGUgSU9IYW5kbGVycyBhbmQgcHJpbnQgdGhlbSBpbiB0aGUgZXJyb3IgbWVzc2FnZS5cbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnVGhlIGN1cnJlbnQgZW52aXJvbm1lbnQgZG9lcyBub3Qgc3VwcG9ydCBsb2NhbCBzdG9yYWdlLicpO1xuICAgIH1cbiAgICB0aGlzLkxTID0gd2luZG93LmxvY2FsU3RvcmFnZTtcblxuICAgIGlmIChtb2RlbFBhdGggPT0gbnVsbCB8fCAhbW9kZWxQYXRoKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgJ0ZvciBsb2NhbCBzdG9yYWdlLCBtb2RlbFBhdGggbXVzdCBub3QgYmUgbnVsbCwgdW5kZWZpbmVkIG9yIGVtcHR5LicpO1xuICAgIH1cbiAgICB0aGlzLm1vZGVsUGF0aCA9IG1vZGVsUGF0aDtcbiAgICB0aGlzLmtleXMgPSBnZXRNb2RlbEtleXModGhpcy5tb2RlbFBhdGgpO1xuICB9XG5cbiAgLyoqXG4gICAqIFNhdmUgbW9kZWwgYXJ0aWZhY3RzIHRvIGJyb3dzZXIgbG9jYWwgc3RvcmFnZS5cbiAgICpcbiAgICogU2VlIHRoZSBkb2N1bWVudGF0aW9uIHRvIGBicm93c2VyTG9jYWxTdG9yYWdlYCBmb3IgZGV0YWlscyBvbiB0aGUgc2F2ZWRcbiAgICogYXJ0aWZhY3RzLlxuICAgKlxuICAgKiBAcGFyYW0gbW9kZWxBcnRpZmFjdHMgVGhlIG1vZGVsIGFydGlmYWN0cyB0byBiZSBzdG9yZWQuXG4gICAqIEByZXR1cm5zIEFuIGluc3RhbmNlIG9mIFNhdmVSZXN1bHQuXG4gICAqL1xuICBhc3luYyBzYXZlKG1vZGVsQXJ0aWZhY3RzOiBNb2RlbEFydGlmYWN0cyk6IFByb21pc2U8U2F2ZVJlc3VsdD4ge1xuICAgIGlmIChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5IGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQnJvd3NlckxvY2FsU3RvcmFnZS5zYXZlKCkgZG9lcyBub3Qgc3VwcG9ydCBzYXZpbmcgbW9kZWwgdG9wb2xvZ3kgJyArXG4gICAgICAgICAgJ2luIGJpbmFyeSBmb3JtYXRzIHlldC4nKTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgdG9wb2xvZ3kgPSBKU09OLnN0cmluZ2lmeShtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KTtcbiAgICAgIGNvbnN0IHdlaWdodFNwZWNzID0gSlNPTi5zdHJpbmdpZnkobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpO1xuXG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0c0luZm86IE1vZGVsQXJ0aWZhY3RzSW5mbyA9XG4gICAgICAgICAgZ2V0TW9kZWxBcnRpZmFjdHNJbmZvRm9ySlNPTihtb2RlbEFydGlmYWN0cyk7XG5cbiAgICAgIHRyeSB7XG4gICAgICAgIHRoaXMuTFMuc2V0SXRlbSh0aGlzLmtleXMuaW5mbywgSlNPTi5zdHJpbmdpZnkobW9kZWxBcnRpZmFjdHNJbmZvKSk7XG4gICAgICAgIHRoaXMuTFMuc2V0SXRlbSh0aGlzLmtleXMudG9wb2xvZ3ksIHRvcG9sb2d5KTtcbiAgICAgICAgdGhpcy5MUy5zZXRJdGVtKHRoaXMua2V5cy53ZWlnaHRTcGVjcywgd2VpZ2h0U3BlY3MpO1xuICAgICAgICB0aGlzLkxTLnNldEl0ZW0oXG4gICAgICAgICAgICB0aGlzLmtleXMud2VpZ2h0RGF0YSxcbiAgICAgICAgICAgIGFycmF5QnVmZmVyVG9CYXNlNjRTdHJpbmcobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSkpO1xuXG4gICAgICAgIC8vIE5vdGUgdGhhdCBKU09OLnN0cmluZ2lmeSBkb2Vzbid0IHdyaXRlIG91dCBrZXlzIHRoYXQgaGF2ZSB1bmRlZmluZWRcbiAgICAgICAgLy8gdmFsdWVzLCBzbyBmb3Igc29tZSBrZXlzLCB3ZSBzZXQgdW5kZWZpbmVkIGluc3RlYWQgb2YgYSBudWxsLWlzaFxuICAgICAgICAvLyB2YWx1ZS5cbiAgICAgICAgY29uc3QgbWV0YWRhdGE6IFJlcXVpcmVkPE1vZGVsTWV0YWRhdGE+ID0ge1xuICAgICAgICAgIGZvcm1hdDogbW9kZWxBcnRpZmFjdHMuZm9ybWF0LFxuICAgICAgICAgIGdlbmVyYXRlZEJ5OiBtb2RlbEFydGlmYWN0cy5nZW5lcmF0ZWRCeSxcbiAgICAgICAgICBjb252ZXJ0ZWRCeTogbW9kZWxBcnRpZmFjdHMuY29udmVydGVkQnksXG4gICAgICAgICAgc2lnbmF0dXJlOiBtb2RlbEFydGlmYWN0cy5zaWduYXR1cmUgIT0gbnVsbCA/XG4gICAgICAgICAgICAgIG1vZGVsQXJ0aWZhY3RzLnNpZ25hdHVyZSA6XG4gICAgICAgICAgICAgIHVuZGVmaW5lZCxcbiAgICAgICAgICB1c2VyRGVmaW5lZE1ldGFkYXRhOiBtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhICE9IG51bGwgP1xuICAgICAgICAgICAgICBtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhIDpcbiAgICAgICAgICAgICAgdW5kZWZpbmVkLFxuICAgICAgICAgIG1vZGVsSW5pdGlhbGl6ZXI6IG1vZGVsQXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIgIT0gbnVsbCA/XG4gICAgICAgICAgICAgIG1vZGVsQXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIgOlxuICAgICAgICAgICAgICB1bmRlZmluZWQsXG4gICAgICAgICAgaW5pdGlhbGl6ZXJTaWduYXR1cmU6IG1vZGVsQXJ0aWZhY3RzLmluaXRpYWxpemVyU2lnbmF0dXJlICE9IG51bGwgP1xuICAgICAgICAgICAgICBtb2RlbEFydGlmYWN0cy5pbml0aWFsaXplclNpZ25hdHVyZSA6XG4gICAgICAgICAgICAgIHVuZGVmaW5lZCxcbiAgICAgICAgICB0cmFpbmluZ0NvbmZpZzogbW9kZWxBcnRpZmFjdHMudHJhaW5pbmdDb25maWcgIT0gbnVsbCA/XG4gICAgICAgICAgICAgIG1vZGVsQXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnIDpcbiAgICAgICAgICAgICAgdW5kZWZpbmVkXG4gICAgICAgIH07XG4gICAgICAgIHRoaXMuTFMuc2V0SXRlbSh0aGlzLmtleXMubW9kZWxNZXRhZGF0YSwgSlNPTi5zdHJpbmdpZnkobWV0YWRhdGEpKTtcblxuICAgICAgICByZXR1cm4ge21vZGVsQXJ0aWZhY3RzSW5mb307XG4gICAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgICAgLy8gSWYgc2F2aW5nIGZhaWxlZCwgY2xlYW4gdXAgYWxsIGl0ZW1zIHNhdmVkIHNvIGZhci5cbiAgICAgICAgcmVtb3ZlSXRlbXModGhpcy5rZXlzKTtcblxuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgRmFpbGVkIHRvIHNhdmUgbW9kZWwgJyR7dGhpcy5tb2RlbFBhdGh9JyB0byBsb2NhbCBzdG9yYWdlOiBgICtcbiAgICAgICAgICAgIGBzaXplIHF1b3RhIGJlaW5nIGV4Y2VlZGVkIGlzIGEgcG9zc2libGUgY2F1c2Ugb2YgdGhpcyBmYWlsdXJlOiBgICtcbiAgICAgICAgICAgIGBtb2RlbFRvcG9sb2d5Qnl0ZXM9JHttb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neUJ5dGVzfSwgYCArXG4gICAgICAgICAgICBgd2VpZ2h0U3BlY3NCeXRlcz0ke21vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzfSwgYCArXG4gICAgICAgICAgICBgd2VpZ2h0RGF0YUJ5dGVzPSR7bW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlc30uYCk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIExvYWQgYSBtb2RlbCBmcm9tIGxvY2FsIHN0b3JhZ2UuXG4gICAqXG4gICAqIFNlZSB0aGUgZG9jdW1lbnRhdGlvbiB0byBgYnJvd3NlckxvY2FsU3RvcmFnZWAgZm9yIGRldGFpbHMgb24gdGhlIHNhdmVkXG4gICAqIGFydGlmYWN0cy5cbiAgICpcbiAgICogQHJldHVybnMgVGhlIGxvYWRlZCBtb2RlbCAoaWYgbG9hZGluZyBzdWNjZWVkcykuXG4gICAqL1xuICBhc3luYyBsb2FkKCk6IFByb21pc2U8TW9kZWxBcnRpZmFjdHM+IHtcbiAgICBjb25zdCBpbmZvID1cbiAgICAgICAgSlNPTi5wYXJzZSh0aGlzLkxTLmdldEl0ZW0odGhpcy5rZXlzLmluZm8pKSBhcyBNb2RlbEFydGlmYWN0c0luZm87XG4gICAgaWYgKGluZm8gPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBJbiBsb2NhbCBzdG9yYWdlLCB0aGVyZSBpcyBubyBtb2RlbCB3aXRoIG5hbWUgJyR7dGhpcy5tb2RlbFBhdGh9J2ApO1xuICAgIH1cblxuICAgIGlmIChpbmZvLm1vZGVsVG9wb2xvZ3lUeXBlICE9PSAnSlNPTicpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnQnJvd3NlckxvY2FsU3RvcmFnZSBkb2VzIG5vdCBzdXBwb3J0IGxvYWRpbmcgbm9uLUpTT04gbW9kZWwgJyArXG4gICAgICAgICAgJ3RvcG9sb2d5IHlldC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCBvdXQ6IE1vZGVsQXJ0aWZhY3RzID0ge307XG5cbiAgICAvLyBMb2FkIHRvcG9sb2d5LlxuICAgIGNvbnN0IHRvcG9sb2d5ID0gSlNPTi5wYXJzZSh0aGlzLkxTLmdldEl0ZW0odGhpcy5rZXlzLnRvcG9sb2d5KSk7XG4gICAgaWYgKHRvcG9sb2d5ID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgSW4gbG9jYWwgc3RvcmFnZSwgdGhlIHRvcG9sb2d5IG9mIG1vZGVsICcke3RoaXMubW9kZWxQYXRofScgYCArXG4gICAgICAgICAgYGlzIG1pc3NpbmcuYCk7XG4gICAgfVxuICAgIG91dC5tb2RlbFRvcG9sb2d5ID0gdG9wb2xvZ3k7XG5cbiAgICAvLyBMb2FkIHdlaWdodCBzcGVjcy5cbiAgICBjb25zdCB3ZWlnaHRTcGVjcyA9IEpTT04ucGFyc2UodGhpcy5MUy5nZXRJdGVtKHRoaXMua2V5cy53ZWlnaHRTcGVjcykpO1xuICAgIGlmICh3ZWlnaHRTcGVjcyA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEluIGxvY2FsIHN0b3JhZ2UsIHRoZSB3ZWlnaHQgc3BlY3Mgb2YgbW9kZWwgJyR7dGhpcy5tb2RlbFBhdGh9JyBgICtcbiAgICAgICAgICBgYXJlIG1pc3NpbmcuYCk7XG4gICAgfVxuICAgIG91dC53ZWlnaHRTcGVjcyA9IHdlaWdodFNwZWNzO1xuXG4gICAgLy8gTG9hZCBtZXRhLWRhdGEgZmllbGRzLlxuICAgIGNvbnN0IG1ldGFkYXRhU3RyaW5nID0gdGhpcy5MUy5nZXRJdGVtKHRoaXMua2V5cy5tb2RlbE1ldGFkYXRhKTtcbiAgICBpZiAobWV0YWRhdGFTdHJpbmcgIT0gbnVsbCkge1xuICAgICAgY29uc3QgbWV0YWRhdGEgPSBKU09OLnBhcnNlKG1ldGFkYXRhU3RyaW5nKSBhcyBNb2RlbE1ldGFkYXRhO1xuICAgICAgb3V0LmZvcm1hdCA9IG1ldGFkYXRhLmZvcm1hdDtcbiAgICAgIG91dC5nZW5lcmF0ZWRCeSA9IG1ldGFkYXRhLmdlbmVyYXRlZEJ5O1xuICAgICAgb3V0LmNvbnZlcnRlZEJ5ID0gbWV0YWRhdGEuY29udmVydGVkQnk7XG4gICAgICBpZiAobWV0YWRhdGEuc2lnbmF0dXJlICE9IG51bGwpIHtcbiAgICAgICAgb3V0LnNpZ25hdHVyZSA9IG1ldGFkYXRhLnNpZ25hdHVyZTtcbiAgICAgIH1cbiAgICAgIGlmIChtZXRhZGF0YS51c2VyRGVmaW5lZE1ldGFkYXRhICE9IG51bGwpIHtcbiAgICAgICAgb3V0LnVzZXJEZWZpbmVkTWV0YWRhdGEgPSBtZXRhZGF0YS51c2VyRGVmaW5lZE1ldGFkYXRhO1xuICAgICAgfVxuICAgICAgaWYgKG1ldGFkYXRhLm1vZGVsSW5pdGlhbGl6ZXIgIT0gbnVsbCkge1xuICAgICAgICBvdXQubW9kZWxJbml0aWFsaXplciA9IG1ldGFkYXRhLm1vZGVsSW5pdGlhbGl6ZXI7XG4gICAgICB9XG4gICAgICBpZiAobWV0YWRhdGEuaW5pdGlhbGl6ZXJTaWduYXR1cmUgIT0gbnVsbCkge1xuICAgICAgICBvdXQuaW5pdGlhbGl6ZXJTaWduYXR1cmUgPSBtZXRhZGF0YS5pbml0aWFsaXplclNpZ25hdHVyZTtcbiAgICAgIH1cbiAgICAgIGlmIChtZXRhZGF0YS50cmFpbmluZ0NvbmZpZyAhPSBudWxsKSB7XG4gICAgICAgIG91dC50cmFpbmluZ0NvbmZpZyA9IG1ldGFkYXRhLnRyYWluaW5nQ29uZmlnO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIExvYWQgd2VpZ2h0IGRhdGEuXG4gICAgY29uc3Qgd2VpZ2h0RGF0YUJhc2U2NCA9IHRoaXMuTFMuZ2V0SXRlbSh0aGlzLmtleXMud2VpZ2h0RGF0YSk7XG4gICAgaWYgKHdlaWdodERhdGFCYXNlNjQgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBJbiBsb2NhbCBzdG9yYWdlLCB0aGUgYmluYXJ5IHdlaWdodCB2YWx1ZXMgb2YgbW9kZWwgYCArXG4gICAgICAgICAgYCcke3RoaXMubW9kZWxQYXRofScgYXJlIG1pc3NpbmcuYCk7XG4gICAgfVxuICAgIG91dC53ZWlnaHREYXRhID0gYmFzZTY0U3RyaW5nVG9BcnJheUJ1ZmZlcih3ZWlnaHREYXRhQmFzZTY0KTtcblxuICAgIHJldHVybiBvdXQ7XG4gIH1cbn1cblxuZXhwb3J0IGNvbnN0IGxvY2FsU3RvcmFnZVJvdXRlcjogSU9Sb3V0ZXIgPSAodXJsOiBzdHJpbmd8c3RyaW5nW10pID0+IHtcbiAgaWYgKCFlbnYoKS5nZXRCb29sKCdJU19CUk9XU0VSJykpIHtcbiAgICByZXR1cm4gbnVsbDtcbiAgfSBlbHNlIHtcbiAgICBpZiAoIUFycmF5LmlzQXJyYXkodXJsKSAmJiB1cmwuc3RhcnRzV2l0aChCcm93c2VyTG9jYWxTdG9yYWdlLlVSTF9TQ0hFTUUpKSB7XG4gICAgICByZXR1cm4gYnJvd3NlckxvY2FsU3RvcmFnZShcbiAgICAgICAgICB1cmwuc2xpY2UoQnJvd3NlckxvY2FsU3RvcmFnZS5VUkxfU0NIRU1FLmxlbmd0aCkpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gbnVsbDtcbiAgICB9XG4gIH1cbn07XG5JT1JvdXRlclJlZ2lzdHJ5LnJlZ2lzdGVyU2F2ZVJvdXRlcihsb2NhbFN0b3JhZ2VSb3V0ZXIpO1xuSU9Sb3V0ZXJSZWdpc3RyeS5yZWdpc3RlckxvYWRSb3V0ZXIobG9jYWxTdG9yYWdlUm91dGVyKTtcblxuLyoqXG4gKiBGYWN0b3J5IGZ1bmN0aW9uIGZvciBsb2NhbCBzdG9yYWdlIElPSGFuZGxlci5cbiAqXG4gKiBUaGlzIGBJT0hhbmRsZXJgIHN1cHBvcnRzIGJvdGggYHNhdmVgIGFuZCBgbG9hZGAuXG4gKlxuICogRm9yIGVhY2ggbW9kZWwncyBzYXZlZCBhcnRpZmFjdHMsIGZvdXIgaXRlbXMgYXJlIHNhdmVkIHRvIGxvY2FsIHN0b3JhZ2UuXG4gKiAgIC0gYCR7UEFUSF9TRVBBUkFUT1J9LyR7bW9kZWxQYXRofS9pbmZvYDogQ29udGFpbnMgbWV0YS1pbmZvIGFib3V0IHRoZVxuICogICAgIG1vZGVsLCBzdWNoIGFzIGRhdGUgc2F2ZWQsIHR5cGUgb2YgdGhlIHRvcG9sb2d5LCBzaXplIGluIGJ5dGVzLCBldGMuXG4gKiAgIC0gYCR7UEFUSF9TRVBBUkFUT1J9LyR7bW9kZWxQYXRofS90b3BvbG9neWA6IE1vZGVsIHRvcG9sb2d5LiBGb3IgS2VyYXMtXG4gKiAgICAgc3R5bGUgbW9kZWxzLCB0aGlzIGlzIGEgc3RyaW5naXplZCBKU09OLlxuICogICAtIGAke1BBVEhfU0VQQVJBVE9SfS8ke21vZGVsUGF0aH0vd2VpZ2h0X3NwZWNzYDogV2VpZ2h0IHNwZWNzIG9mIHRoZVxuICogICAgIG1vZGVsLCBjYW4gYmUgdXNlZCB0byBkZWNvZGUgdGhlIHNhdmVkIGJpbmFyeSB3ZWlnaHQgdmFsdWVzIChzZWVcbiAqICAgICBpdGVtIGJlbG93KS5cbiAqICAgLSBgJHtQQVRIX1NFUEFSQVRPUn0vJHttb2RlbFBhdGh9L3dlaWdodF9kYXRhYDogQ29uY2F0ZW5hdGVkIGJpbmFyeVxuICogICAgIHdlaWdodCB2YWx1ZXMsIHN0b3JlZCBhcyBhIGJhc2U2NC1lbmNvZGVkIHN0cmluZy5cbiAqXG4gKiBTYXZpbmcgbWF5IHRocm93IGFuIGBFcnJvcmAgaWYgdGhlIHRvdGFsIHNpemUgb2YgdGhlIGFydGlmYWN0cyBleGNlZWQgdGhlXG4gKiBicm93c2VyLXNwZWNpZmljIHF1b3RhLlxuICpcbiAqIEBwYXJhbSBtb2RlbFBhdGggQSB1bmlxdWUgaWRlbnRpZmllciBmb3IgdGhlIG1vZGVsIHRvIGJlIHNhdmVkLiBNdXN0IGJlIGFcbiAqICAgbm9uLWVtcHR5IHN0cmluZy5cbiAqIEByZXR1cm5zIEFuIGluc3RhbmNlIG9mIGBJT0hhbmRsZXJgLCB3aGljaCBjYW4gYmUgdXNlZCB3aXRoLCBlLmcuLFxuICogICBgdGYuTW9kZWwuc2F2ZWAuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBicm93c2VyTG9jYWxTdG9yYWdlKG1vZGVsUGF0aDogc3RyaW5nKTogSU9IYW5kbGVyIHtcbiAgcmV0dXJuIG5ldyBCcm93c2VyTG9jYWxTdG9yYWdlKG1vZGVsUGF0aCk7XG59XG5cbmV4cG9ydCBjbGFzcyBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlciBpbXBsZW1lbnRzIE1vZGVsU3RvcmVNYW5hZ2VyIHtcbiAgcHJpdmF0ZSByZWFkb25seSBMUzogU3RvcmFnZTtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICBhc3NlcnQoXG4gICAgICAgIGVudigpLmdldEJvb2woJ0lTX0JST1dTRVInKSxcbiAgICAgICAgKCkgPT4gJ0N1cnJlbnQgZW52aXJvbm1lbnQgaXMgbm90IGEgd2ViIGJyb3dzZXInKTtcbiAgICBhc3NlcnQoXG4gICAgICAgIHR5cGVvZiB3aW5kb3cgPT09ICd1bmRlZmluZWQnIHx8XG4gICAgICAgICAgICB0eXBlb2Ygd2luZG93LmxvY2FsU3RvcmFnZSAhPT0gJ3VuZGVmaW5lZCcsXG4gICAgICAgICgpID0+ICdDdXJyZW50IGJyb3dzZXIgZG9lcyBub3QgYXBwZWFyIHRvIHN1cHBvcnQgbG9jYWxTdG9yYWdlJyk7XG4gICAgdGhpcy5MUyA9IHdpbmRvdy5sb2NhbFN0b3JhZ2U7XG4gIH1cblxuICBhc3luYyBsaXN0TW9kZWxzKCk6IFByb21pc2U8e1twYXRoOiBzdHJpbmddOiBNb2RlbEFydGlmYWN0c0luZm99PiB7XG4gICAgY29uc3Qgb3V0OiB7W3BhdGg6IHN0cmluZ106IE1vZGVsQXJ0aWZhY3RzSW5mb30gPSB7fTtcbiAgICBjb25zdCBwcmVmaXggPSBQQVRIX1BSRUZJWCArIFBBVEhfU0VQQVJBVE9SO1xuICAgIGNvbnN0IHN1ZmZpeCA9IFBBVEhfU0VQQVJBVE9SICsgSU5GT19TVUZGSVg7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLkxTLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCBrZXkgPSB0aGlzLkxTLmtleShpKTtcbiAgICAgIGlmIChrZXkuc3RhcnRzV2l0aChwcmVmaXgpICYmIGtleS5lbmRzV2l0aChzdWZmaXgpKSB7XG4gICAgICAgIGNvbnN0IG1vZGVsUGF0aCA9IGdldE1vZGVsUGF0aEZyb21LZXkoa2V5KTtcbiAgICAgICAgb3V0W21vZGVsUGF0aF0gPSBKU09OLnBhcnNlKHRoaXMuTFMuZ2V0SXRlbShrZXkpKSBhcyBNb2RlbEFydGlmYWN0c0luZm87XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBvdXQ7XG4gIH1cblxuICBhc3luYyByZW1vdmVNb2RlbChwYXRoOiBzdHJpbmcpOiBQcm9taXNlPE1vZGVsQXJ0aWZhY3RzSW5mbz4ge1xuICAgIHBhdGggPSBtYXliZVN0cmlwU2NoZW1lKHBhdGgpO1xuICAgIGNvbnN0IGtleXMgPSBnZXRNb2RlbEtleXMocGF0aCk7XG4gICAgaWYgKHRoaXMuTFMuZ2V0SXRlbShrZXlzLmluZm8pID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgQ2Fubm90IGZpbmQgbW9kZWwgYXQgcGF0aCAnJHtwYXRofSdgKTtcbiAgICB9XG4gICAgY29uc3QgaW5mbyA9IEpTT04ucGFyc2UodGhpcy5MUy5nZXRJdGVtKGtleXMuaW5mbykpIGFzIE1vZGVsQXJ0aWZhY3RzSW5mbztcbiAgICByZW1vdmVJdGVtcyhrZXlzKTtcbiAgICByZXR1cm4gaW5mbztcbiAgfVxufVxuIl19