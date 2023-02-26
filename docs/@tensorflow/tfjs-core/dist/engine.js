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
import { KernelBackend } from './backends/backend';
import { Environment, setEnvironmentGlobal } from './environment';
import { getGlobalNamespace } from './global_util';
import { Add, Cast, Identity } from './kernel_names';
import { getGradient, getKernel, getKernelsForBackend } from './kernel_registry';
import * as log from './log';
import { Profiler } from './profiler';
import { backpropagateGradients, getFilteredNodesXToY } from './tape';
import { setTensorTracker, Tensor, Variable } from './tensor';
import { getTensorsInContainer } from './tensor_util';
import * as util from './util';
import { bytesFromStringArray, makeOnesTypedArray, now, sizeFromShape } from './util';
function isRegisteredKernelInvocation(kernelInvocation) {
    return kernelInvocation.kernelName != null;
}
class EngineState {
    constructor() {
        // Public since optimizers will use it.
        this.registeredVariables = {};
        this.nextTapeNodeId = 0;
        this.numBytes = 0;
        this.numTensors = 0;
        this.numStringTensors = 0;
        this.numDataBuffers = 0;
        // Number of nested tf.grad() statements when computing higher-order
        // gradients. E.g. `1` for first-order gradients and `2` for second-order
        // gradients. Used to track if the tape should be removed after a backprop.
        this.gradientDepth = 0;
        // Number of nested kernel calls. When kernel depth is greater than 1, we turn
        // off the tape.
        this.kernelDepth = 0;
        this.scopeStack = [];
        /**
         * Keeps track of the number of data moves during a kernel execution. We
         * maintain a stack since kernels can call other kernels, recursively.
         */
        this.numDataMovesStack = [];
        this.nextScopeId = 0;
        this.tensorInfo = new WeakMap();
        this.profiling = false;
        this.activeProfile = {
            newBytes: 0,
            newTensors: 0,
            peakBytes: 0,
            kernels: [],
            result: null,
            get kernelNames() {
                return Array.from(new Set(this.kernels.map(k => k.name)));
            }
        };
    }
    dispose() {
        for (const variableName in this.registeredVariables) {
            this.registeredVariables[variableName].dispose();
        }
    }
}
export class Engine {
    constructor(ENV) {
        this.ENV = ENV;
        this.registry = {};
        this.registryFactory = {};
        this.pendingBackendInitId = 0;
        this.state = new EngineState();
    }
    async ready() {
        if (this.pendingBackendInit != null) {
            return this.pendingBackendInit.then(() => { });
        }
        if (this.backendInstance != null) {
            return;
        }
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const success = await this.initializeBackend(backendName).success;
            if (success) {
                await this.setBackend(backendName);
                return;
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    get backend() {
        if (this.pendingBackendInit != null) {
            throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make ` +
                `sure to await tf.ready() or await tf.setBackend() before calling ` +
                `other methods`);
        }
        if (this.backendInstance == null) {
            const { name, asyncInit } = this.initializeBackendsAndReturnBest();
            if (asyncInit) {
                throw new Error(`The highest priority backend '${name}' has not yet been ` +
                    `initialized. Make sure to await tf.ready() or ` +
                    `await tf.setBackend() before calling other methods`);
            }
            this.setBackend(name);
        }
        return this.backendInstance;
    }
    backendNames() {
        return Object.keys(this.registryFactory);
    }
    findBackend(backendName) {
        if (!(backendName in this.registry)) {
            // If the backend hasn't been initialized but we have a registry entry for
            // it, initialize it and return it.
            if (backendName in this.registryFactory) {
                const { asyncInit } = this.initializeBackend(backendName);
                if (asyncInit) {
                    // Backend is not ready yet.
                    return null;
                }
            }
            else {
                return null;
            }
        }
        return this.registry[backendName];
    }
    findBackendFactory(backendName) {
        if (!(backendName in this.registryFactory)) {
            return null;
        }
        return this.registryFactory[backendName].factory;
    }
    registerBackend(backendName, factory, priority = 1) {
        if (backendName in this.registryFactory) {
            log.warn(`${backendName} backend was already registered. ` +
                `Reusing existing backend factory.`);
            return false;
        }
        this.registryFactory[backendName] = { factory, priority };
        return true;
    }
    async setBackend(backendName) {
        if (this.registryFactory[backendName] == null) {
            throw new Error(`Backend name '${backendName}' not found in registry`);
        }
        this.backendName = backendName;
        if (this.registry[backendName] == null) {
            this.backendInstance = null;
            const { success, asyncInit } = this.initializeBackend(backendName);
            const result = asyncInit ? await success : success;
            if (!result) {
                return false;
            }
        }
        this.backendInstance = this.registry[backendName];
        this.setupRegisteredKernels();
        // Reset the profiler.
        this.profiler = new Profiler(this.backendInstance);
        return true;
    }
    setupRegisteredKernels() {
        const kernels = getKernelsForBackend(this.backendName);
        kernels.forEach(kernel => {
            if (kernel.setupFunc != null) {
                kernel.setupFunc(this.backendInstance);
            }
        });
    }
    disposeRegisteredKernels(backendName) {
        const kernels = getKernelsForBackend(backendName);
        kernels.forEach(kernel => {
            if (kernel.disposeFunc != null) {
                kernel.disposeFunc(this.registry[backendName]);
            }
        });
    }
    /**
     * Initializes a backend by looking up the backend name in the factory
     * registry and calling the factory method. Returns a boolean representing
     * whether the initialization of the backend suceeded. Throws an error if
     * there is no backend in the factory registry.
     */
    initializeBackend(backendName) {
        const registryFactoryEntry = this.registryFactory[backendName];
        if (registryFactoryEntry == null) {
            throw new Error(`Cannot initialize backend ${backendName}, no registration found.`);
        }
        try {
            const backend = registryFactoryEntry.factory();
            /* Test if the factory returns a promise.
            Done in a more liberal way than
            previous 'Promise.resolve(backend)===backend'
            as we needed to account for custom Promise
            implementations (e.g. Angular) */
            if (backend && !(backend instanceof KernelBackend) &&
                typeof backend.then === 'function') {
                const promiseId = ++this.pendingBackendInitId;
                const success = backend
                    .then(backendInstance => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.registry[backendName] = backendInstance;
                    this.pendingBackendInit = null;
                    return true;
                })
                    .catch(err => {
                    // Outdated promise. Another backend was set in the meantime.
                    if (promiseId < this.pendingBackendInitId) {
                        return false;
                    }
                    this.pendingBackendInit = null;
                    log.warn(`Initialization of backend ${backendName} failed`);
                    log.warn(err.stack || err.message);
                    return false;
                });
                this.pendingBackendInit = success;
                return { success, asyncInit: true };
            }
            else {
                this.registry[backendName] = backend;
                return { success: true, asyncInit: false };
            }
        }
        catch (err) {
            log.warn(`Initialization of backend ${backendName} failed`);
            log.warn(err.stack || err.message);
            return { success: false, asyncInit: false };
        }
    }
    removeBackend(backendName) {
        if (!(backendName in this.registryFactory)) {
            throw new Error(`${backendName} backend not found in registry`);
        }
        if (this.backendName === backendName && this.pendingBackendInit != null) {
            // There is a pending promise of the backend we want to remove. Make it
            // obsolete.
            this.pendingBackendInitId++;
        }
        if (backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        delete this.registryFactory[backendName];
        // Unset the backend if it is active.
        if (this.backendName === backendName) {
            this.pendingBackendInit = null;
            this.backendName = null;
            this.backendInstance = null;
        }
    }
    getSortedBackends() {
        if (Object.keys(this.registryFactory).length === 0) {
            throw new Error('No backend found in registry.');
        }
        return Object.keys(this.registryFactory).sort((a, b) => {
            // Highest priority comes first.
            return this.registryFactory[b].priority -
                this.registryFactory[a].priority;
        });
    }
    initializeBackendsAndReturnBest() {
        const sortedBackends = this.getSortedBackends();
        for (let i = 0; i < sortedBackends.length; i++) {
            const backendName = sortedBackends[i];
            const { success, asyncInit } = this.initializeBackend(backendName);
            if (asyncInit || success) {
                return { name: backendName, asyncInit };
            }
        }
        throw new Error(`Could not initialize any backends, all backend initializations ` +
            `failed.`);
    }
    moveData(backend, dataId) {
        const info = this.state.tensorInfo.get(dataId);
        const srcBackend = info.backend;
        const values = this.readSync(dataId);
        const refCount = srcBackend.refCount(dataId);
        // Delete the tensor from the old backend and move it to the new
        // backend.
        srcBackend.disposeData(dataId, true);
        info.backend = backend;
        backend.move(dataId, values, info.shape, info.dtype, refCount);
        if (this.shouldCheckForMemLeaks()) {
            // Track the number of moves during a kernel execution to correctly
            // detect memory leaks.
            this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1]++;
        }
    }
    tidy(nameOrFn, fn) {
        let name = null;
        if (fn == null) {
            // Called with only 1 argument.
            if (typeof nameOrFn !== 'function') {
                throw new Error('Please provide a function to tidy()');
            }
            fn = nameOrFn;
        }
        else {
            // Called with 2 arguments.
            if (typeof nameOrFn !== 'string' && !(nameOrFn instanceof String)) {
                throw new Error('When calling with two arguments, the first argument ' +
                    'to tidy() must be a string');
            }
            if (typeof fn !== 'function') {
                throw new Error('When calling with two arguments, the 2nd argument ' +
                    'to tidy() must be a function');
            }
            name = nameOrFn;
            // TODO(nsthorat,smilkov): Do operation logging and performance
            // profiling.
        }
        let result;
        return this.scopedRun(() => this.startScope(name), () => this.endScope(result), () => {
            result = fn();
            if (result instanceof Promise) {
                console.error('Cannot return a Promise inside of tidy.');
            }
            return result;
        });
    }
    scopedRun(start, end, f) {
        start();
        try {
            const res = f();
            end();
            return res;
        }
        catch (ex) {
            end();
            throw ex;
        }
    }
    nextTensorId() {
        return Engine.nextTensorId++;
    }
    nextVariableId() {
        return Engine.nextVariableId++;
    }
    /**
     * This method is called instead of the public-facing tensor.clone() when
     * saving a tensor for backwards pass. It makes sure to add the clone
     * operation to the tape regardless of being called inside a kernel
     * execution.
     */
    clone(x) {
        const y = ENGINE.runKernel(Identity, { x });
        const inputs = { x };
        const grad = (dy) => ({
            x: () => {
                const dtype = 'float32';
                const gradInputs = { x: dy };
                const attrs = { dtype };
                return ENGINE.runKernel(Cast, gradInputs, 
                // tslint:disable-next-line: no-unnecessary-type-assertion
                attrs);
            }
        });
        const saved = [];
        this.addTapeNode(this.state.activeScope.name, inputs, [y], grad, saved, {});
        return y;
    }
    /**
     * Execute a kernel with the given name and return the output tensor.
     *
     * @param kernelName The name of the kernel to execute.
     * @param inputs A map of input names to tensors.
     * @param attrs A map of attribute names to their values. An attribute is a
     *     primitive (non-tensor) input to the kernel.
     * @param inputsToSave A list of tensors, inputs to save for the backprop
     *     computation.
     * @param outputsToSave A list of booleans, specifying which output to save
     *     for the backprop computation. These are booleans since the output
     * tensors are not visible to the user.
     */
    runKernel(kernelName, inputs, attrs) {
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        const hasKernel = getKernel(kernelName, this.backendName) != null;
        if (!hasKernel) {
            throw new Error(`Kernel '${kernelName}' not registered for backend '${this.backendName}'`);
        }
        return this.runKernelFunc({ kernelName, inputs, attrs });
    }
    shouldCheckForMemLeaks() {
        return this.ENV.getBool('IS_TEST');
    }
    checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos) {
        const numDataIdsAfter = this.backend.numDataIds();
        // Count the number of data ids associated with the result of the kernel.
        let numOutputDataIds = 0;
        outInfos.forEach(info => {
            // Complex numbers allocate 3 data ids, one for 'real', one for
            // 'imaginary', and one for the container that holds the former two.
            numOutputDataIds += (info.dtype === 'complex64' ? 3 : 1);
        });
        // Account for the number of moves during kernel execution. A "data move"
        // can happen in the middle of a kernel execution, placing a new (key,value)
        // pair in the data storage. Since data moves have net zero effect (we
        // always remove the data from the old backend), we have to cancel them out
        // when detecting memory leaks.
        const numMoves = this.state.numDataMovesStack[this.state.numDataMovesStack.length - 1];
        const dataIdsLeaked = numDataIdsAfter - numDataIdsBefore - numOutputDataIds - numMoves;
        if (dataIdsLeaked > 0) {
            throw new Error(`Backend '${this.backendName}' has an internal memory leak ` +
                `(${dataIdsLeaked} data ids) after running '${kernelName}'`);
        }
    }
    /**
     * Internal helper method to execute a kernel Func
     *
     * Use `runKernel` to execute kernels from outside of engine.
     */
    runKernelFunc(kernelParams) {
        let outputs;
        let saved = [];
        const isTapeOn = this.isTapeOn();
        const startingBytecount = this.state.numBytes;
        const startingNumTensors = this.state.numTensors;
        if (this.shouldCheckForMemLeaks()) {
            this.state.numDataMovesStack.push(0);
        }
        let kernelFunc;
        if (this.backendName == null) {
            // backend has not been initialized yet (backend initialization is lazy
            // can be deferred until an op/ kernel is run).
            // The below getter has side effects that will try to initialize the
            // backend and set properties like this.backendName
            // tslint:disable-next-line: no-unused-expression
            this.backend;
        }
        let out;
        const kernelOrScopeName = isRegisteredKernelInvocation(kernelParams) ?
            kernelParams.kernelName :
            this.state.activeScope != null ? this.state.activeScope.name : '';
        // Create the kernelFunc from either a registered kernel OR passed in
        // forward/backward functions (used by custom grad). In this context a
        // kernelFunc wraps a kernel implementation with some bookkeeping.
        if (isRegisteredKernelInvocation(kernelParams)) {
            const { kernelName, inputs, attrs } = kernelParams;
            if (this.backendName == null) {
                // backend has not been initialized yet (backend initialization is lazy
                // can be deferred until an op/ kernel is run).
                // The below getter has side effects that will try to initialize the
                // backend and set properties like this.backendName
                // tslint:disable-next-line: no-unused-expression
                this.backend;
            }
            const kernel = getKernel(kernelName, this.backendName);
            util.assert(kernel != null, () => `Cannot find registered kernel '${kernelName}' for backend '${this.backendName}'`);
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = kernel.kernelFunc({ inputs, attrs, backend: this.backend });
                const outInfos = Array.isArray(out) ? out : [out];
                if (this.shouldCheckForMemLeaks()) {
                    this.checkKernelForMemLeak(kernelName, numDataIdsBefore, outInfos);
                }
                const outTensors = outInfos.map((outInfo) => {
                    // todo (yassogba) remove this option (Tensor) when node backend
                    // methods have been modularized and they all return tensorInfo.
                    // TensorInfos do not have a rank attribute.
                    if (outInfo.rank != null) {
                        return outInfo;
                    }
                    return this.makeTensorFromTensorInfo(outInfo);
                });
                // Save any required inputs and outputs.
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since there would be no backprop for these tensors
                // (which would otherwise dispose them).
                if (isTapeOn) {
                    const tensorsToSave = this.getTensorsForGradient(kernelName, inputs, outTensors);
                    saved = this.saveTensorsForBackwardMode(tensorsToSave);
                }
                return outTensors;
            };
        }
        else {
            const { forwardFunc } = kernelParams;
            // Running a customGrad op.
            const saveFunc = (tensors) => {
                // Do not save unless we are recording to the tape. Otherwise it would
                // cause a mem leak since we would never run backprop, which disposes
                // the kept tensors.
                if (!isTapeOn) {
                    return;
                }
                saved = tensors.map(tensor => this.keep(this.clone(tensor)));
            };
            kernelFunc = () => {
                const numDataIdsBefore = this.backend.numDataIds();
                out = this.tidy(() => forwardFunc(this.backend, saveFunc));
                const outs = (Array.isArray(out) ? out : [out]);
                if (this.shouldCheckForMemLeaks()) {
                    // Scope name is used to print a more helpful error message if needed.
                    this.checkKernelForMemLeak(kernelOrScopeName, numDataIdsBefore, outs);
                }
                return outs;
            };
        }
        //
        // Run the kernelFunc. Optionally profiling it.
        //
        const { inputs, attrs } = kernelParams;
        const backwardsFunc = isRegisteredKernelInvocation(kernelParams) ?
            null :
            kernelParams.backwardsFunc;
        let kernelProfile;
        this.scopedRun(
        // Stop recording to a tape when running a kernel.
        () => this.state.kernelDepth++, () => this.state.kernelDepth--, () => {
            if (!this.ENV.getBool('DEBUG') && !this.state.profiling) {
                outputs = kernelFunc();
            }
            else {
                kernelProfile = this.profiler.profileKernel(kernelOrScopeName, inputs, () => kernelFunc());
                if (this.ENV.getBool('DEBUG')) {
                    this.profiler.logKernelProfile(kernelProfile);
                }
                outputs = kernelProfile.outputs;
            }
        });
        if (isTapeOn) {
            this.addTapeNode(kernelOrScopeName, inputs, outputs, backwardsFunc, saved, attrs);
        }
        if (this.state.profiling) {
            this.state.activeProfile.kernels.push({
                name: kernelOrScopeName,
                bytesAdded: this.state.numBytes - startingBytecount,
                totalBytesSnapshot: this.state.numBytes,
                tensorsAdded: this.state.numTensors - startingNumTensors,
                totalTensorsSnapshot: this.state.numTensors,
                inputShapes: Object.keys(inputs).map(key => inputs[key] != null ? inputs[key].shape : null),
                outputShapes: outputs.map(item => item.shape),
                kernelTimeMs: kernelProfile.timeMs,
                extraInfo: kernelProfile.extraInfo
            });
        }
        return (Array.isArray(out) ? outputs : outputs[0]);
    }
    /**
     * Saves tensors used in forward mode for use in backward mode.
     *
     * @param tensors the list of tensors to save.
     */
    saveTensorsForBackwardMode(tensors) {
        const saved = tensors.map(tensor => this.keep(this.clone(tensor)));
        return saved;
    }
    /**
     * Returns a list of tensors to save for a given gradient calculation.
     *
     * @param kernelName name of kernel to look up gradient for.
     * @param inputs a map of input tensors.
     * @param outputs an array of output tensors from forward mode of kernel.
     */
    getTensorsForGradient(kernelName, inputs, outputs) {
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            const inputsToSave = gradConfig.inputsToSave || [];
            const outputsToSave = gradConfig.outputsToSave || [];
            // If saveAllInputs is true, all inputs will be saved. Otherwise, inputs
            // specified in inputsToSave will be saved.
            let inputTensorsToSave;
            if (gradConfig.saveAllInputs) {
                util.assert(Array.isArray(inputs), () => 'saveAllInputs is true, expected inputs to be an array.');
                inputTensorsToSave = Object.keys(inputs).map((key) => inputs[key]);
            }
            else {
                inputTensorsToSave = inputsToSave.map((inputName) => inputs[inputName]);
            }
            const outputTensorsToSave = outputs.filter((_, i) => outputsToSave[i]);
            return inputTensorsToSave.concat(outputTensorsToSave);
        }
        // We return an empty list rather than throw an error because the kernel we
        // are looking up may not actually be relevant to backproping through the
        // overall function
        //
        // See 'does not error if irrelevant (pruned) ops are missing grads' test
        // in gradients_test.ts for an example.
        return [];
    }
    /**
     * Internal method used by public APIs for tensor creation. Makes a new
     * tensor with the provided shape, dtype and values. It always
     * creates a new data id and writes the values to the underlying backend.
     */
    makeTensor(values, shape, dtype, backend) {
        if (values == null) {
            throw new Error('Values passed to engine.makeTensor() are null');
        }
        dtype = dtype || 'float32';
        backend = backend || this.backend;
        let backendVals = values;
        if (dtype === 'string' && util.isString(values[0])) {
            backendVals = values.map(d => util.encodeString(d));
        }
        const dataId = backend.write(backendVals, shape, dtype);
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        // Count bytes for string tensors.
        if (dtype === 'string') {
            const info = this.state.tensorInfo.get(dataId);
            const newBytes = bytesFromStringArray(backendVals);
            this.state.numBytes += newBytes - info.bytes;
            info.bytes = newBytes;
        }
        return t;
    }
    /**
     * Internal method used by backends. Makes a new tensor
     * that is a wrapper around an existing data id. It doesn't create
     * a new data id, only increments the ref count used in memory tracking.
     * @deprecated
     */
    makeTensorFromDataId(dataId, shape, dtype, backend) {
        dtype = dtype || 'float32';
        const tensorInfo = { dataId, shape, dtype };
        return this.makeTensorFromTensorInfo(tensorInfo, backend);
    }
    /**
     * Internal method used by backends. Makes a new tensor that is a wrapper
     * around an existing data id in TensorInfo. It doesn't create a new data id,
     * only increments the ref count used in memory tracking.
     */
    makeTensorFromTensorInfo(tensorInfo, backend) {
        const { dataId, shape, dtype } = tensorInfo;
        const t = new Tensor(shape, dtype, dataId, this.nextTensorId());
        this.trackTensor(t, backend);
        return t;
    }
    makeVariable(initialValue, trainable = true, name, dtype) {
        name = name || this.nextVariableId().toString();
        if (dtype != null && dtype !== initialValue.dtype) {
            initialValue = initialValue.cast(dtype);
        }
        const v = new Variable(initialValue, trainable, name, this.nextTensorId());
        if (this.state.registeredVariables[v.name] != null) {
            throw new Error(`Variable with name ${v.name} was already registered`);
        }
        this.state.registeredVariables[v.name] = v;
        this.incRef(v, this.backend);
        return v;
    }
    trackTensor(a, backend) {
        this.state.numTensors++;
        if (a.dtype === 'string') {
            this.state.numStringTensors++;
        }
        // Bytes for complex numbers are counted by their components. Bytes for
        // string tensors are counted when writing values.
        let bytes = 0;
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            bytes = a.size * util.bytesPerElement(a.dtype);
        }
        this.state.numBytes += bytes;
        if (!this.state.tensorInfo.has(a.dataId)) {
            this.state.numDataBuffers++;
            this.state.tensorInfo.set(a.dataId, {
                backend: backend || this.backend,
                dtype: a.dtype,
                shape: a.shape,
                bytes
            });
        }
        if (!(a instanceof Variable)) {
            this.track(a);
        }
    }
    // Track the tensor by dataId and increase the refCount for the dataId in the
    // backend.
    // TODO(pyu10055): This is currently used by makeVariable method, to increase
    // refCount on the backend for the dataId. It can potentially be replaced with
    // Identity op indead of calling backend directly.
    incRef(a, backend) {
        this.trackTensor(a, backend);
        this.backend.incRef(a.dataId);
    }
    removeDataId(dataId, backend) {
        if (this.state.tensorInfo.has(dataId) &&
            this.state.tensorInfo.get(dataId).backend === backend) {
            this.state.tensorInfo.delete(dataId);
            this.state.numDataBuffers--;
        }
    }
    disposeTensor(a) {
        if (!this.state.tensorInfo.has(a.dataId)) {
            return;
        }
        const info = this.state.tensorInfo.get(a.dataId);
        this.state.numTensors--;
        if (a.dtype === 'string') {
            this.state.numStringTensors--;
            this.state.numBytes -= info.bytes;
        }
        // Don't count bytes for complex numbers as they are counted by their
        // components.
        if (a.dtype !== 'complex64' && a.dtype !== 'string') {
            const bytes = a.size * util.bytesPerElement(a.dtype);
            this.state.numBytes -= bytes;
        }
        // Remove the reference to dataId if backend dispose the data successfully
        if (info.backend.disposeData(a.dataId)) {
            this.removeDataId(a.dataId, info.backend);
        }
        // TODO(nsthorat): Construct an error and save the stack trace for
        // debugging when in debug mode. Creating a stack trace is too expensive
        // to do unconditionally.
    }
    disposeVariables() {
        for (const varName in this.state.registeredVariables) {
            const v = this.state.registeredVariables[varName];
            this.disposeVariable(v);
        }
    }
    disposeVariable(v) {
        this.disposeTensor(v);
        if (this.state.registeredVariables[v.name] != null) {
            delete this.state.registeredVariables[v.name];
        }
    }
    memory() {
        const info = this.backend.memory();
        info.numTensors = this.state.numTensors;
        info.numDataBuffers = this.state.numDataBuffers;
        info.numBytes = this.state.numBytes;
        if (this.state.numStringTensors > 0) {
            info.unreliable = true;
            if (info.reasons == null) {
                info.reasons = [];
            }
            info.reasons.push('Memory usage by string tensors is approximate ' +
                '(2 bytes per character)');
        }
        return info;
    }
    async profile(query) {
        this.state.profiling = true;
        const startBytes = this.state.numBytes;
        const startNumTensors = this.state.numTensors;
        this.state.activeProfile.kernels = [];
        this.state.activeProfile.result = await query();
        this.state.profiling = false;
        this.state.activeProfile.peakBytes = Math.max(...this.state.activeProfile.kernels.map(d => d.totalBytesSnapshot));
        this.state.activeProfile.newBytes = this.state.numBytes - startBytes;
        this.state.activeProfile.newTensors =
            this.state.numTensors - startNumTensors;
        for (const kernel of this.state.activeProfile.kernels) {
            kernel.kernelTimeMs = await kernel.kernelTimeMs;
            kernel.extraInfo = await kernel.extraInfo;
        }
        return this.state.activeProfile;
    }
    isTapeOn() {
        return this.state.gradientDepth > 0 && this.state.kernelDepth === 0;
    }
    addTapeNode(kernelName, inputs, outputs, gradientsFunc, saved, attrs) {
        const tapeNode = { id: this.state.nextTapeNodeId++, kernelName, inputs, outputs, saved };
        const gradConfig = getGradient(kernelName);
        if (gradConfig != null) {
            gradientsFunc = gradConfig.gradFunc;
        }
        if (gradientsFunc != null) {
            tapeNode.gradient = (dys) => {
                // TODO(smilkov): To optimize back-prop, pass dys that are not used in
                // the backprop graph to the user as null instead of zeros
                dys = dys.map((dy, i) => {
                    if (dy == null) {
                        const output = outputs[i];
                        const vals = util.makeZerosTypedArray(output.size, output.dtype);
                        return this.makeTensor(vals, output.shape, output.dtype);
                    }
                    return dy;
                });
                // Grad functions of ops with single outputs expect a dy, while ops
                // with multiple outputs expect dys (array of dy).
                return gradientsFunc(dys.length > 1 ? dys : dys[0], saved, attrs);
            };
        }
        this.state.activeTape.push(tapeNode);
    }
    keep(result) {
        result.kept = true;
        return result;
    }
    startTape() {
        if (this.state.gradientDepth === 0) {
            this.state.activeTape = [];
        }
        this.state.gradientDepth++;
    }
    endTape() {
        this.state.gradientDepth--;
    }
    /**
     * Start a scope. Use this with endScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    startScope(name) {
        const scopeInfo = {
            track: [],
            name: 'unnamed scope',
            id: this.state.nextScopeId++
        };
        if (name) {
            scopeInfo.name = name;
        }
        this.state.scopeStack.push(scopeInfo);
        this.state.activeScope = scopeInfo;
    }
    /**
     * End a scope. Use this with startScope() to achieve the same functionality
     * as scope() without the need for a function closure.
     */
    endScope(result) {
        const tensorsToTrackInParent = getTensorsInContainer(result);
        const tensorsToTrackInParentSet = new Set(tensorsToTrackInParent.map(t => t.id));
        // Dispose the arrays tracked in this scope.
        for (let i = 0; i < this.state.activeScope.track.length; i++) {
            const tensor = this.state.activeScope.track[i];
            if (!tensor.kept && !tensorsToTrackInParentSet.has(tensor.id)) {
                tensor.dispose();
            }
        }
        const oldScope = this.state.scopeStack.pop();
        this.state.activeScope = this.state.scopeStack.length === 0 ?
            null :
            this.state.scopeStack[this.state.scopeStack.length - 1];
        // Track the current result in the parent scope.
        tensorsToTrackInParent.forEach(tensor => {
            // Only track the tensor if was allocated in the inner scope and is not
            // globally kept.
            if (!tensor.kept && tensor.scopeId === oldScope.id) {
                this.track(tensor);
            }
        });
    }
    /**
     * Returns gradients of `f` with respect to each of the `xs`. The gradients
     * returned are of the same length as `xs`, but some might be null if `f`
     * was not a function of that `x`. It also takes optional dy to multiply the
     * gradient, which defaults to `1`.
     */
    gradients(f, xs, dy, allowNoGradients = false) {
        util.assert(xs.length > 0, () => 'gradients() received an empty list of xs.');
        if (dy != null && dy.dtype !== 'float32') {
            throw new Error(`dy must have 'float32' dtype, but has '${dy.dtype}'`);
        }
        const y = this.scopedRun(() => this.startTape(), () => this.endTape(), () => this.tidy('forward', f));
        util.assert(y instanceof Tensor, () => 'The result y returned by f() must be a tensor.');
        // Filter out the nodes that don't connect x => y.
        const filteredTape = getFilteredNodesXToY(this.state.activeTape, xs, y);
        if (!allowNoGradients && filteredTape.length === 0 && xs.length > 0) {
            throw new Error('Cannot compute gradient of y=f(x) with respect to x. Make sure ' +
                'that the f you passed encloses all operations that lead from x ' +
                'to y.');
        }
        return this.tidy('backward', () => {
            const accumulatedGradientMap = {};
            accumulatedGradientMap[y.id] = (dy == null) ? ones(y.shape) : dy;
            // Backprop gradients through the filtered nodes.
            backpropagateGradients(accumulatedGradientMap, filteredTape, 
            // Pass the tidy function to avoid circular dep with `tape.ts`.
            f => this.tidy(f), 
            // Pass an add function to avoide a circular dep with `tape.ts`.
            add);
            const grads = xs.map(x => accumulatedGradientMap[x.id]);
            if (this.state.gradientDepth === 0) {
                // This means that we are not computing higher-order gradients
                // and can clean up the tape.
                this.state.activeTape.forEach(node => {
                    for (const tensor of node.saved) {
                        tensor.dispose();
                    }
                });
                this.state.activeTape = null;
            }
            return { value: y, grads };
        });
    }
    customGrad(f) {
        util.assert(util.isFunction(f), () => 'The f passed in customGrad(f) must be a function.');
        return (...inputs) => {
            util.assert(inputs.every(t => t instanceof Tensor), () => 'The args passed in customGrad(f)(x1, x2,...) must all be ' +
                'tensors');
            let res;
            const inputMap = {};
            inputs.forEach((input, i) => {
                inputMap[i] = input;
            });
            const forwardFunc = (_, save) => {
                res = f(...[...inputs, save]);
                util.assert(res.value instanceof Tensor, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.value` is a tensor');
                util.assert(util.isFunction(res.gradFunc), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function.');
                return res.value;
            };
            const backwardsFunc = (dy, saved) => {
                const gradRes = res.gradFunc(dy, saved);
                const grads = Array.isArray(gradRes) ? gradRes : [gradRes];
                util.assert(grads.length === inputs.length, () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'the same number of tensors as inputs passed to f(...).');
                util.assert(grads.every(t => t instanceof Tensor), () => 'The function f passed in customGrad(f) must return an ' +
                    'object where `obj.gradFunc` is a function that returns ' +
                    'a list of only tensors.');
                const gradMap = {};
                grads.forEach((grad, i) => {
                    gradMap[i] = () => grad;
                });
                return gradMap;
            };
            return this.runKernelFunc({
                forwardFunc,
                backwardsFunc,
                inputs: inputMap,
            });
        };
    }
    readSync(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readSync(dataId);
    }
    read(dataId) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.read(dataId);
    }
    readToGPU(dataId, options) {
        // Route the read to the correct backend.
        const info = this.state.tensorInfo.get(dataId);
        return info.backend.readToGPU(dataId, options);
    }
    async time(query) {
        const start = now();
        const timingInfo = await this.backend.time(query);
        timingInfo.wallMs = now() - start;
        return timingInfo;
    }
    /**
     * Tracks a Tensor in the current scope to be automatically cleaned up
     * when the current scope ends, and returns the value.
     *
     * @param result The Tensor to track in the current scope.
     */
    track(result) {
        if (this.state.activeScope != null) {
            result.scopeId = this.state.activeScope.id;
            this.state.activeScope.track.push(result);
        }
        return result;
    }
    get registeredVariables() {
        return this.state.registeredVariables;
    }
    /**
     * Resets the engine state. Removes all backends but does not remove
     * registered backend factories.
     */
    reset() {
        // Make any pending promise obsolete.
        this.pendingBackendInitId++;
        this.state.dispose();
        this.ENV.reset();
        this.state = new EngineState();
        for (const backendName in this.registry) {
            this.disposeRegisteredKernels(backendName);
            this.registry[backendName].dispose();
            delete this.registry[backendName];
        }
        this.backendName = null;
        this.backendInstance = null;
        this.pendingBackendInit = null;
    }
}
Engine.nextTensorId = 0;
Engine.nextVariableId = 0;
function ones(shape) {
    const values = makeOnesTypedArray(sizeFromShape(shape), 'float32');
    return ENGINE.makeTensor(values, shape, 'float32');
}
export function getOrMakeEngine() {
    const ns = getGlobalNamespace();
    if (ns._tfengine == null) {
        const environment = new Environment(ns);
        ns._tfengine = new Engine(environment);
    }
    setEnvironmentGlobal(ns._tfengine.ENV);
    // Tell the current tensor interface that the global engine is responsible
    // for tracking.
    setTensorTracker(() => ns._tfengine);
    return ns._tfengine;
}
export const ENGINE = getOrMakeEngine();
/**
 * A implementation of the add op for use within engine and tape.
 *
 * This allows us to avoid a circular dependency between add.ts and engine.
 * It is exported to be available in tape tests.
 */
export function add(a, b) {
    // We duplicate Add here to avoid a circular dependency with add.ts.
    const inputs = { a, b };
    return ENGINE.runKernel(Add, inputs);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW5naW5lLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9lbmdpbmUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUErQixhQUFhLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUMvRSxPQUFPLEVBQUMsV0FBVyxFQUFFLG9CQUFvQixFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQ2hFLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUNqRCxPQUFPLEVBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxRQUFRLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNuRCxPQUFPLEVBQUUsV0FBVyxFQUFFLFNBQVMsRUFBRSxvQkFBb0IsRUFBMEIsTUFBTSxtQkFBbUIsQ0FBQztBQUV6RyxPQUFPLEtBQUssR0FBRyxNQUFNLE9BQU8sQ0FBQztBQUM3QixPQUFPLEVBQWdCLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNuRCxPQUFPLEVBQUMsc0JBQXNCLEVBQUUsb0JBQW9CLEVBQVcsTUFBTSxRQUFRLENBQUM7QUFDOUUsT0FBTyxFQUE0QixnQkFBZ0IsRUFBRSxNQUFNLEVBQWlCLFFBQVEsRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUd0RyxPQUFPLEVBQUMscUJBQXFCLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFFcEQsT0FBTyxLQUFLLElBQUksTUFBTSxRQUFRLENBQUM7QUFDL0IsT0FBTyxFQUFDLG9CQUFvQixFQUFFLGtCQUFrQixFQUFFLEdBQUcsRUFBRSxhQUFhLEVBQUMsTUFBTSxRQUFRLENBQUM7QUF1RXBGLFNBQVMsNEJBQTRCLENBRWpDLGdCQUNnQztJQUVsQyxPQUFRLGdCQUFrRCxDQUFDLFVBQVUsSUFBSSxJQUFJLENBQUM7QUFDaEYsQ0FBQztBQUVELE1BQU0sV0FBVztJQUFqQjtRQUNFLHVDQUF1QztRQUN2Qyx3QkFBbUIs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eEJwQyxTQUFTLElBQUksQ0FBQyxLQUFlO0lBQzNCLE1BQU0sTUFBTSxHQUFHLGtCQUFrQixDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUNuRSxPQUFPLE1BQU0sQ0FBQyxVQUFVLENBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxTQUFTLENBQUMsQ0FBQztBQUNyRCxDQUFDO0FBRUQsTUFBTSxVQUFVLGVBQWU7SUFDN0IsTUFBTSxFQUFFLEdBQUcsa0JBQWtCLEVBQW9DLENBQUM7SUFDbEUsSUFBSSxFQUFFLENBQUMsU0FBUyxJQUFJLElBQUksRUFBRTtRQUN4QixNQUFNLFdBQVcsR0FBRyxJQUFJLFdBQVcsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUN4QyxFQUFFLENBQUMsU0FBUyxHQUFHLElBQUksTUFBTSxDQUFDLFdBQVcsQ0FBQyxDQUFDO0tBQ3hDO0lBQ0Qsb0JBQW9CLENBQUMsRUFBRSxDQUFDLFNBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUV2QywwRUFBMEU7SUFDMUUsZ0JBQWdCO0lBQ2hCLGdCQUFnQixDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUNyQyxPQUFPLEVBQUUsQ0FBQyxTQUFTLENBQUM7QUFDdEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxlQUFlLEVBQUUsQ0FBQztBQUV4Qzs7Ozs7R0FLRztBQUNILE1BQU0sVUFBVSxHQUFHLENBQUMsQ0FBUyxFQUFFLENBQVM7SUFDdEMsb0VBQW9FO0lBQ3BFLE1BQU0sTUFBTSxHQUFHLEVBQUMsQ0FBQyxFQUFFLENBQUMsRUFBQyxDQUFDO0lBQ3RCLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxHQUFHLEVBQUUsTUFBbUMsQ0FBQyxDQUFDO0FBQ3BFLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7QmFja2VuZFRpbWluZ0luZm8sIERhdGFNb3ZlciwgS2VybmVsQmFja2VuZH0gZnJvbSAnLi9iYWNrZW5kcy9iYWNrZW5kJztcbmltcG9ydCB7RW52aXJvbm1lbnQsIHNldEVudmlyb25tZW50R2xvYmFsfSBmcm9tICcuL2Vudmlyb25tZW50JztcbmltcG9ydCB7Z2V0R2xvYmFsTmFtZXNwYWNlfSBmcm9tICcuL2dsb2JhbF91dGlsJztcbmltcG9ydCB7QWRkLCBDYXN0LCBJZGVudGl0eX0gZnJvbSAnLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHsgZ2V0R3JhZGllbnQsIGdldEtlcm5lbCwgZ2V0S2VybmVsc0ZvckJhY2tlbmQsIEdyYWRGdW5jLCBOYW1lZEF0dHJNYXAgfSBmcm9tICcuL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQgeyBUZW5zb3JJbmZvIH0gZnJvbSAnLi90ZW5zb3JfaW5mbyc7XG5pbXBvcnQgKiBhcyBsb2cgZnJvbSAnLi9sb2cnO1xuaW1wb3J0IHtLZXJuZWxQcm9maWxlLCBQcm9maWxlcn0gZnJvbSAnLi9wcm9maWxlcic7XG5pbXBvcnQge2JhY2twcm9wYWdhdGVHcmFkaWVudHMsIGdldEZpbHRlcmVkTm9kZXNYVG9ZLCBUYXBlTm9kZX0gZnJvbSAnLi90YXBlJztcbmltcG9ydCB7RGF0YVRvR1BVT3B0aW9ucywgR1BVRGF0YSwgc2V0VGVuc29yVHJhY2tlciwgVGVuc29yLCBUZW5zb3JUcmFja2VyLCBWYXJpYWJsZX0gZnJvbSAnLi90ZW5zb3InO1xuaW1wb3J0IHtEYXRhSWR9IGZyb20gJy4vdGVuc29yX2luZm8nO1xuaW1wb3J0IHtHcmFkU2F2ZUZ1bmMsIE5hbWVkVGVuc29yTWFwLCBOYW1lZFZhcmlhYmxlTWFwLCBUZW5zb3JDb250YWluZXJ9IGZyb20gJy4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Z2V0VGVuc29yc0luQ29udGFpbmVyfSBmcm9tICcuL3RlbnNvcl91dGlsJztcbmltcG9ydCB7QmFja2VuZFZhbHVlcywgRGF0YVR5cGUsIERhdGFWYWx1ZXN9IGZyb20gJy4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuL3V0aWwnO1xuaW1wb3J0IHtieXRlc0Zyb21TdHJpbmdBcnJheSwgbWFrZU9uZXNUeXBlZEFycmF5LCBub3csIHNpemVGcm9tU2hhcGV9IGZyb20gJy4vdXRpbCc7XG5cbi8qKlxuICogQSBmdW5jdGlvbiB0aGF0IGNvbXB1dGVzIGFuIG91dHB1dC4gVGhlIHNhdmUgZnVuY3Rpb24gaXMgZm9yIHNhdmluZyB0ZW5zb3JzXG4gKiBjb21wdXRlZCBpbiB0aGUgZm9yd2FyZCBwYXNzLCB0aGF0IHdlIG5lZWQgaW4gdGhlIGJhY2t3YXJkIHBhc3MuXG4gKi9cbmV4cG9ydCB0eXBlIEZvcndhcmRGdW5jPFQ+ID0gKGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQsIHNhdmU/OiBHcmFkU2F2ZUZ1bmMpID0+IFQ7XG5cbi8qKlxuICogQGRvY2FsaWFzIChhOiBUZW5zb3IsIGI6IFRlbnNvciwuLi4sIHNhdmU/OiBGdW5jdGlvbikgPT4ge1xuICogICB2YWx1ZTogVGVuc29yLFxuICogICBncmFkRnVuYzogKGR5OiBUZW5zb3IsIHNhdmVkPzogTmFtZWRUZW5zb3JNYXApID0+IFRlbnNvciB8IFRlbnNvcltdXG4gKiB9XG4gKi9cbmV4cG9ydCB0eXBlIEN1c3RvbUdyYWRpZW50RnVuYzxUIGV4dGVuZHMgVGVuc29yPiA9XG4gICAgKC4uLmlucHV0czogQXJyYXk8VGVuc29yfEdyYWRTYXZlRnVuYz4pID0+IHtcbiAgICAgIHZhbHVlOiBUO1xuICAgICAgZ3JhZEZ1bmM6IChkeTogVCwgc2F2ZWQ6IFRlbnNvcltdKSA9PiBUZW5zb3IgfCBUZW5zb3JbXTtcbiAgICB9O1xuXG5leHBvcnQgdHlwZSBNZW1vcnlJbmZvID0ge1xuICBudW1UZW5zb3JzOiBudW1iZXI7IG51bURhdGFCdWZmZXJzOiBudW1iZXI7IG51bUJ5dGVzOiBudW1iZXI7XG4gIHVucmVsaWFibGU/OiBib29sZWFuOyByZWFzb25zOiBzdHJpbmdbXTtcbn07XG5cbnR5cGUgS2VybmVsSW5mbyA9IHtcbiAgbmFtZTogc3RyaW5nOyBieXRlc0FkZGVkOiBudW1iZXI7IHRvdGFsQnl0ZXNTbmFwc2hvdDogbnVtYmVyO1xuICB0ZW5zb3JzQWRkZWQ6IG51bWJlcjtcbiAgdG90YWxUZW5zb3JzU25hcHNob3Q6IG51bWJlcjtcbiAgaW5wdXRTaGFwZXM6IG51bWJlcltdW107XG4gIG91dHB1dFNoYXBlczogbnVtYmVyW11bXTtcbiAga2VybmVsVGltZU1zOiBudW1iZXIgfCB7ZXJyb3I6IHN0cmluZ30gfCBQcm9taXNlPG51bWJlcnx7ZXJyb3I6IHN0cmluZ30+O1xuICBleHRyYUluZm86IHN0cmluZyB8IFByb21pc2U8c3RyaW5nPjtcbn07XG5cbmV4cG9ydCB0eXBlIFByb2ZpbGVJbmZvID0ge1xuICBuZXdCeXRlczogbnVtYmVyOyBuZXdUZW5zb3JzOiBudW1iZXI7IHBlYWtCeXRlczogbnVtYmVyO1xuICBrZXJuZWxzOiBLZXJuZWxJbmZvW107XG4gIHJlc3VsdDogVGVuc29yQ29udGFpbmVyO1xuICBrZXJuZWxOYW1lczogc3RyaW5nW107XG59O1xuXG5leHBvcnQgaW50ZXJmYWNlIFRpbWluZ0luZm8gZXh0ZW5kcyBCYWNrZW5kVGltaW5nSW5mbyB7XG4gIHdhbGxNczogbnVtYmVyO1xufVxuXG4vKiogQGRvY2FsaWFzIEZ1bmN0aW9uICovXG5leHBvcnQgdHlwZSBTY29wZUZuPFQgZXh0ZW5kcyBUZW5zb3JDb250YWluZXI+ID0gKCkgPT4gVDtcblxuaW50ZXJmYWNlIFNjb3BlU3RhdGUge1xuICB0cmFjazogVGVuc29yW107XG4gIG5hbWU6IHN0cmluZztcbiAgaWQ6IG51bWJlcjtcbn1cblxuaW50ZXJmYWNlIFJlZ2lzdGVyZWRLZXJuZWxJbnZvY2F0aW9uPEkgZXh0ZW5kcyBOYW1lZFRlbnNvck1hcD4ge1xuICBrZXJuZWxOYW1lOiBzdHJpbmc7XG4gIGlucHV0czogSTtcbiAgYXR0cnM/OiBOYW1lZEF0dHJNYXA7XG59XG5cbmludGVyZmFjZSBDdXN0b21HcmFkS2VybmVsSW52b2NhdGlvbjxUIGV4dGVuZHMgVGVuc29yfFRlbnNvcltdLFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJIGV4dGVuZHMgTmFtZWRUZW5zb3JNYXA+IHtcbiAgZm9yd2FyZEZ1bmM6IEZvcndhcmRGdW5jPFQ+O1xuICBiYWNrd2FyZHNGdW5jOiAoZHk6IFQsIHNhdmVkOiBUZW5zb3JbXSkgPT4ge1xuICAgIFtQIGluIGtleW9mIEldOiAoKSA9PiBJW1BdXG4gIH07XG4gIGlucHV0czogSTtcbiAgYXR0cnM/OiBOYW1lZEF0dHJNYXA7XG59XG5cbmZ1bmN0aW9uIGlzUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb248VCBleHRlbmRzIFRlbnNvcnxUZW5zb3JbXSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEkgZXh0ZW5kcyBOYW1lZFRlbnNvck1hcD4oXG4gICAga2VybmVsSW52b2NhdGlvbjogUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb248ST58XG4gICAgQ3VzdG9tR3JhZEtlcm5lbEludm9jYXRpb248VCwgST4pOlxuICAgIGtlcm5lbEludm9jYXRpb24gaXMgUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb248ST4ge1xuICByZXR1cm4gKGtlcm5lbEludm9jYXRpb24gYXMgUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb248ST4pLmtlcm5lbE5hbWUgIT0gbnVsbDtcbn1cblxuY2xhc3MgRW5naW5lU3RhdGUge1xuICAvLyBQdWJsaWMgc2luY2Ugb3B0aW1pemVycyB3aWxsIHVzZSBpdC5cbiAgcmVnaXN0ZXJlZFZhcmlhYmxlczogTmFtZWRWYXJpYWJsZU1hcCA9IHt9O1xuXG4gIG5leHRUYXBlTm9kZUlkID0gMDtcbiAgbnVtQnl0ZXMgPSAwO1xuICBudW1UZW5zb3JzID0gMDtcbiAgbnVtU3RyaW5nVGVuc29ycyA9IDA7XG4gIG51bURhdGFCdWZmZXJzID0gMDtcblxuICBhY3RpdmVUYXBlOiBUYXBlTm9kZVtdO1xuICAvLyBOdW1iZXIgb2YgbmVzdGVkIHRmLmdyYWQoKSBzdGF0ZW1lbnRzIHdoZW4gY29tcHV0aW5nIGhpZ2hlci1vcmRlclxuICAvLyBncmFkaWVudHMuIEUuZy4gYDFgIGZvciBmaXJzdC1vcmRlciBncmFkaWVudHMgYW5kIGAyYCBmb3Igc2Vjb25kLW9yZGVyXG4gIC8vIGdyYWRpZW50cy4gVXNlZCB0byB0cmFjayBpZiB0aGUgdGFwZSBzaG91bGQgYmUgcmVtb3ZlZCBhZnRlciBhIGJhY2twcm9wLlxuICBncmFkaWVudERlcHRoID0gMDtcbiAgLy8gTnVtYmVyIG9mIG5lc3RlZCBrZXJuZWwgY2FsbHMuIFdoZW4ga2VybmVsIGRlcHRoIGlzIGdyZWF0ZXIgdGhhbiAxLCB3ZSB0dXJuXG4gIC8vIG9mZiB0aGUgdGFwZS5cbiAga2VybmVsRGVwdGggPSAwO1xuXG4gIC8vIEtlZXAgVGVuc29ycyB0aGF0IHBhcmFsbGVsIHRoZSB0YXBlcy5cbiAgYWN0aXZlU2NvcGU6IFNjb3BlU3RhdGU7XG4gIHNjb3BlU3RhY2s6IFNjb3BlU3RhdGVbXSA9IFtdO1xuICAvKipcbiAgICogS2VlcHMgdHJhY2sgb2YgdGhlIG51bWJlciBvZiBkYXRhIG1vdmVzIGR1cmluZyBhIGtlcm5lbCBleGVjdXRpb24uIFdlXG4gICAqIG1haW50YWluIGEgc3RhY2sgc2luY2Uga2VybmVscyBjYW4gY2FsbCBvdGhlciBrZXJuZWxzLCByZWN1cnNpdmVseS5cbiAgICovXG4gIG51bURhdGFNb3Zlc1N0YWNrOiBudW1iZXJbXSA9IFtdO1xuICBuZXh0U2NvcGVJZCA9IDA7XG5cbiAgdGVuc29ySW5mbyA9IG5ldyBXZWFrTWFwPERhdGFJZCwge1xuICAgIGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQsXG4gICAgYnl0ZXM6IG51bWJlcixcbiAgICBkdHlwZTogRGF0YVR5cGUsXG4gICAgc2hhcGU6IG51bWJlcltdXG4gIH0+KCk7XG5cbiAgcHJvZmlsaW5nID0gZmFsc2U7XG4gIGFjdGl2ZVByb2ZpbGU6IFByb2ZpbGVJbmZvID0ge1xuICAgIG5ld0J5dGVzOiAwLFxuICAgIG5ld1RlbnNvcnM6IDAsXG4gICAgcGVha0J5dGVzOiAwLFxuICAgIGtlcm5lbHM6IFtdLFxuICAgIHJlc3VsdDogbnVsbCxcbiAgICBnZXQga2VybmVsTmFtZXMoKTpcbiAgICAgICAgc3RyaW5nW10ge1xuICAgICAgICAgIHJldHVybiBBcnJheS5mcm9tKG5ldyBTZXQodGhpcy5rZXJuZWxzLm1hcChrID0+IGsubmFtZSkpKTtcbiAgICAgICAgfVxuICB9O1xuXG4gIGRpc3Bvc2UoKSB7XG4gICAgZm9yIChjb25zdCB2YXJpYWJsZU5hbWUgaW4gdGhpcy5yZWdpc3RlcmVkVmFyaWFibGVzKSB7XG4gICAgICB0aGlzLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdmFyaWFibGVOYW1lXS5kaXNwb3NlKCk7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBFbmdpbmUgaW1wbGVtZW50cyBUZW5zb3JUcmFja2VyLCBEYXRhTW92ZXIge1xuICBzdGF0ZTogRW5naW5lU3RhdGU7XG4gIGJhY2tlbmROYW1lOiBzdHJpbmc7XG4gIHJlZ2lzdHJ5OiB7W2lkOiBzdHJpbmddOiBLZXJuZWxCYWNrZW5kfSA9IHt9O1xuICByZWdpc3RyeUZhY3Rvcnk6IHtcbiAgICBbaWQ6IHN0cmluZ106IHtcbiAgICAgIGZhY3Rvcnk6ICgpID0+IEtlcm5lbEJhY2tlbmQgfCBQcm9taXNlPEtlcm5lbEJhY2tlbmQ+LFxuICAgICAgcHJpb3JpdHk6IG51bWJlclxuICAgIH1cbiAgfSA9IHt9O1xuXG4gIHByaXZhdGUgcHJvZmlsZXI6IFByb2ZpbGVyO1xuICBwcml2YXRlIGJhY2tlbmRJbnN0YW5jZTogS2VybmVsQmFja2VuZDtcbiAgcHJpdmF0ZSBwZW5kaW5nQmFja2VuZEluaXQ6IFByb21pc2U8Ym9vbGVhbj47XG4gIHByaXZhdGUgcGVuZGluZ0JhY2tlbmRJbml0SWQgPSAwO1xuXG4gIGNvbnN0cnVjdG9yKHB1YmxpYyBFTlY6IEVudmlyb25tZW50KSB7XG4gICAgdGhpcy5zdGF0ZSA9IG5ldyBFbmdpbmVTdGF0ZSgpO1xuICB9XG5cbiAgYXN5bmMgcmVhZHkoKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgaWYgKHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ICE9IG51bGwpIHtcbiAgICAgIHJldHVybiB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdC50aGVuKCgpID0+IHt9KTtcbiAgICB9XG4gICAgaWYgKHRoaXMuYmFja2VuZEluc3RhbmNlICE9IG51bGwpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3Qgc29ydGVkQmFja2VuZHMgPSB0aGlzLmdldFNvcnRlZEJhY2tlbmRzKCk7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNvcnRlZEJhY2tlbmRzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBiYWNrZW5kTmFtZSA9IHNvcnRlZEJhY2tlbmRzW2ldO1xuICAgICAgY29uc3Qgc3VjY2VzcyA9IGF3YWl0IHRoaXMuaW5pdGlhbGl6ZUJhY2tlbmQoYmFja2VuZE5hbWUpLnN1Y2Nlc3M7XG4gICAgICBpZiAoc3VjY2Vzcykge1xuICAgICAgICBhd2FpdCB0aGlzLnNldEJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgQ291bGQgbm90IGluaXRpYWxpemUgYW55IGJhY2tlbmRzLCBhbGwgYmFja2VuZCBpbml0aWFsaXphdGlvbnMgYCArXG4gICAgICAgIGBmYWlsZWQuYCk7XG4gIH1cblxuICBnZXQgYmFja2VuZCgpOiBLZXJuZWxCYWNrZW5kIHtcbiAgICBpZiAodGhpcy5wZW5kaW5nQmFja2VuZEluaXQgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBCYWNrZW5kICcke3RoaXMuYmFja2VuZE5hbWV9JyBoYXMgbm90IHlldCBiZWVuIGluaXRpYWxpemVkLiBNYWtlIGAgK1xuICAgICAgICAgIGBzdXJlIHRvIGF3YWl0IHRmLnJlYWR5KCkgb3IgYXdhaXQgdGYuc2V0QmFja2VuZCgpIGJlZm9yZSBjYWxsaW5nIGAgK1xuICAgICAgICAgIGBvdGhlciBtZXRob2RzYCk7XG4gICAgfVxuICAgIGlmICh0aGlzLmJhY2tlbmRJbnN0YW5jZSA9PSBudWxsKSB7XG4gICAgICBjb25zdCB7bmFtZSwgYXN5bmNJbml0fSA9IHRoaXMuaW5pdGlhbGl6ZUJhY2tlbmRzQW5kUmV0dXJuQmVzdCgpO1xuICAgICAgaWYgKGFzeW5jSW5pdCkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICBgVGhlIGhpZ2hlc3QgcHJpb3JpdHkgYmFja2VuZCAnJHtuYW1lfScgaGFzIG5vdCB5ZXQgYmVlbiBgICtcbiAgICAgICAgICAgIGBpbml0aWFsaXplZC4gTWFrZSBzdXJlIHRvIGF3YWl0IHRmLnJlYWR5KCkgb3IgYCArXG4gICAgICAgICAgICBgYXdhaXQgdGYuc2V0QmFja2VuZCgpIGJlZm9yZSBjYWxsaW5nIG90aGVyIG1ldGhvZHNgKTtcbiAgICAgIH1cbiAgICAgIHRoaXMuc2V0QmFja2VuZChuYW1lKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuYmFja2VuZEluc3RhbmNlO1xuICB9XG5cbiAgYmFja2VuZE5hbWVzKCk6IHN0cmluZ1tdIHtcbiAgICByZXR1cm4gT2JqZWN0LmtleXModGhpcy5yZWdpc3RyeUZhY3RvcnkpO1xuICB9XG5cbiAgZmluZEJhY2tlbmQoYmFja2VuZE5hbWU6IHN0cmluZyk6IEtlcm5lbEJhY2tlbmQge1xuICAgIGlmICghKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnkpKSB7XG4gICAgICAvLyBJZiB0aGUgYmFja2VuZCBoYXNuJ3QgYmVlbiBpbml0aWFsaXplZCBidXQgd2UgaGF2ZSBhIHJlZ2lzdHJ5IGVudHJ5IGZvclxuICAgICAgLy8gaXQsIGluaXRpYWxpemUgaXQgYW5kIHJldHVybiBpdC5cbiAgICAgIGlmIChiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkge1xuICAgICAgICBjb25zdCB7YXN5bmNJbml0fSA9IHRoaXMuaW5pdGlhbGl6ZUJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgICAgICBpZiAoYXN5bmNJbml0KSB7XG4gICAgICAgICAgLy8gQmFja2VuZCBpcyBub3QgcmVhZHkgeWV0LlxuICAgICAgICAgIHJldHVybiBudWxsO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gbnVsbDtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdO1xuICB9XG5cbiAgZmluZEJhY2tlbmRGYWN0b3J5KGJhY2tlbmROYW1lOiBzdHJpbmcpOlxuICAgICAgKCkgPT4gS2VybmVsQmFja2VuZCB8IFByb21pc2U8S2VybmVsQmFja2VuZD4ge1xuICAgIGlmICghKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnlGYWN0b3J5KSkge1xuICAgICAgcmV0dXJuIG51bGw7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiYWNrZW5kTmFtZV0uZmFjdG9yeTtcbiAgfVxuXG4gIHJlZ2lzdGVyQmFja2VuZChcbiAgICAgIGJhY2tlbmROYW1lOiBzdHJpbmcsXG4gICAgICBmYWN0b3J5OiAoKSA9PiBLZXJuZWxCYWNrZW5kIHwgUHJvbWlzZTxLZXJuZWxCYWNrZW5kPixcbiAgICAgIHByaW9yaXR5ID0gMSk6IGJvb2xlYW4ge1xuICAgIGlmIChiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkge1xuICAgICAgbG9nLndhcm4oXG4gICAgICAgICAgYCR7YmFja2VuZE5hbWV9IGJhY2tlbmQgd2FzIGFscmVhZHkgcmVnaXN0ZXJlZC4gYCArXG4gICAgICAgICAgYFJldXNpbmcgZXhpc3RpbmcgYmFja2VuZCBmYWN0b3J5LmApO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgICB0aGlzLnJlZ2lzdHJ5RmFjdG9yeVtiYWNrZW5kTmFtZV0gPSB7ZmFjdG9yeSwgcHJpb3JpdHl9O1xuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgYXN5bmMgc2V0QmFja2VuZChiYWNrZW5kTmFtZTogc3RyaW5nKTogUHJvbWlzZTxib29sZWFuPiB7XG4gICAgaWYgKHRoaXMucmVnaXN0cnlGYWN0b3J5W2JhY2tlbmROYW1lXSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEJhY2tlbmQgbmFtZSAnJHtiYWNrZW5kTmFtZX0nIG5vdCBmb3VuZCBpbiByZWdpc3RyeWApO1xuICAgIH1cbiAgICB0aGlzLmJhY2tlbmROYW1lID0gYmFja2VuZE5hbWU7XG4gICAgaWYgKHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdID09IG51bGwpIHtcbiAgICAgIHRoaXMuYmFja2VuZEluc3RhbmNlID0gbnVsbDtcbiAgICAgIGNvbnN0IHtzdWNjZXNzLCBhc3luY0luaXR9ID0gdGhpcy5pbml0aWFsaXplQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAgICBjb25zdCByZXN1bHQgPSBhc3luY0luaXQgPyBhd2FpdCBzdWNjZXNzIDogc3VjY2VzcztcbiAgICAgIGlmICghcmVzdWx0KSB7XG4gICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG4gICAgdGhpcy5iYWNrZW5kSW5zdGFuY2UgPSB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXTtcbiAgICB0aGlzLnNldHVwUmVnaXN0ZXJlZEtlcm5lbHMoKTtcbiAgICAvLyBSZXNldCB0aGUgcHJvZmlsZXIuXG4gICAgdGhpcy5wcm9maWxlciA9IG5ldyBQcm9maWxlcih0aGlzLmJhY2tlbmRJbnN0YW5jZSk7XG5cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIHByaXZhdGUgc2V0dXBSZWdpc3RlcmVkS2VybmVscygpOiB2b2lkIHtcbiAgICBjb25zdCBrZXJuZWxzID0gZ2V0S2VybmVsc0ZvckJhY2tlbmQodGhpcy5iYWNrZW5kTmFtZSk7XG4gICAga2VybmVscy5mb3JFYWNoKGtlcm5lbCA9PiB7XG4gICAgICBpZiAoa2VybmVsLnNldHVwRnVuYyAhPSBudWxsKSB7XG4gICAgICAgIGtlcm5lbC5zZXR1cEZ1bmModGhpcy5iYWNrZW5kSW5zdGFuY2UpO1xuICAgICAgfVxuICAgIH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBkaXNwb3NlUmVnaXN0ZXJlZEtlcm5lbHMoYmFja2VuZE5hbWU6IHN0cmluZyk6IHZvaWQge1xuICAgIGNvbnN0IGtlcm5lbHMgPSBnZXRLZXJuZWxzRm9yQmFja2VuZChiYWNrZW5kTmFtZSk7XG4gICAga2VybmVscy5mb3JFYWNoKGtlcm5lbCA9PiB7XG4gICAgICBpZiAoa2VybmVsLmRpc3Bvc2VGdW5jICE9IG51bGwpIHtcbiAgICAgICAga2VybmVsLmRpc3Bvc2VGdW5jKHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbml0aWFsaXplcyBhIGJhY2tlbmQgYnkgbG9va2luZyB1cCB0aGUgYmFja2VuZCBuYW1lIGluIHRoZSBmYWN0b3J5XG4gICAqIHJlZ2lzdHJ5IGFuZCBjYWxsaW5nIHRoZSBmYWN0b3J5IG1ldGhvZC4gUmV0dXJucyBhIGJvb2xlYW4gcmVwcmVzZW50aW5nXG4gICAqIHdoZXRoZXIgdGhlIGluaXRpYWxpemF0aW9uIG9mIHRoZSBiYWNrZW5kIHN1Y2VlZGVkLiBUaHJvd3MgYW4gZXJyb3IgaWZcbiAgICogdGhlcmUgaXMgbm8gYmFja2VuZCBpbiB0aGUgZmFjdG9yeSByZWdpc3RyeS5cbiAgICovXG4gIHByaXZhdGUgaW5pdGlhbGl6ZUJhY2tlbmQoYmFja2VuZE5hbWU6IHN0cmluZyk6XG4gICAgICB7c3VjY2VzczogYm9vbGVhbnxQcm9taXNlPGJvb2xlYW4+LCBhc3luY0luaXQ6IGJvb2xlYW59IHtcbiAgICBjb25zdCByZWdpc3RyeUZhY3RvcnlFbnRyeSA9IHRoaXMucmVnaXN0cnlGYWN0b3J5W2JhY2tlbmROYW1lXTtcbiAgICBpZiAocmVnaXN0cnlGYWN0b3J5RW50cnkgPT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBDYW5ub3QgaW5pdGlhbGl6ZSBiYWNrZW5kICR7YmFja2VuZE5hbWV9LCBubyByZWdpc3RyYXRpb24gZm91bmQuYCk7XG4gICAgfVxuXG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IGJhY2tlbmQgPSByZWdpc3RyeUZhY3RvcnlFbnRyeS5mYWN0b3J5KCk7XG4gICAgICAvKiBUZXN0IGlmIHRoZSBmYWN0b3J5IHJldHVybnMgYSBwcm9taXNlLlxuICAgICAgRG9uZSBpbiBhIG1vcmUgbGliZXJhbCB3YXkgdGhhblxuICAgICAgcHJldmlvdXMgJ1Byb21pc2UucmVzb2x2ZShiYWNrZW5kKT09PWJhY2tlbmQnXG4gICAgICBhcyB3ZSBuZWVkZWQgdG8gYWNjb3VudCBmb3IgY3VzdG9tIFByb21pc2VcbiAgICAgIGltcGxlbWVudGF0aW9ucyAoZS5nLiBBbmd1bGFyKSAqL1xuICAgICAgaWYgKGJhY2tlbmQgJiYgIShiYWNrZW5kIGluc3RhbmNlb2YgS2VybmVsQmFja2VuZCkgJiZcbiAgICAgICAgICB0eXBlb2YgYmFja2VuZC50aGVuID09PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgIGNvbnN0IHByb21pc2VJZCA9ICsrdGhpcy5wZW5kaW5nQmFja2VuZEluaXRJZDtcbiAgICAgICAgY29uc3Qgc3VjY2VzcyA9XG4gICAgICAgICAgICBiYWNrZW5kXG4gICAgICAgICAgICAgICAgLnRoZW4oYmFja2VuZEluc3RhbmNlID0+IHtcbiAgICAgICAgICAgICAgICAgIC8vIE91dGRhdGVkIHByb21pc2UuIEFub3RoZXIgYmFja2VuZCB3YXMgc2V0IGluIHRoZSBtZWFudGltZS5cbiAgICAgICAgICAgICAgICAgIGlmIChwcm9taXNlSWQgPCB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdElkKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdID0gYmFja2VuZEluc3RhbmNlO1xuICAgICAgICAgICAgICAgICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgPSBudWxsO1xuICAgICAgICAgICAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICAgICAgICAgICAgfSlcbiAgICAgICAgICAgICAgICAuY2F0Y2goZXJyID0+IHtcbiAgICAgICAgICAgICAgICAgIC8vIE91dGRhdGVkIHByb21pc2UuIEFub3RoZXIgYmFja2VuZCB3YXMgc2V0IGluIHRoZSBtZWFudGltZS5cbiAgICAgICAgICAgICAgICAgIGlmIChwcm9taXNlSWQgPCB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdElkKSB7XG4gICAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ID0gbnVsbDtcbiAgICAgICAgICAgICAgICAgIGxvZy53YXJuKGBJbml0aWFsaXphdGlvbiBvZiBiYWNrZW5kICR7YmFja2VuZE5hbWV9IGZhaWxlZGApO1xuICAgICAgICAgICAgICAgICAgbG9nLndhcm4oZXJyLnN0YWNrIHx8IGVyci5tZXNzYWdlKTtcbiAgICAgICAgICAgICAgICAgIHJldHVybiBmYWxzZTtcbiAgICAgICAgICAgICAgICB9KTtcbiAgICAgICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgPSBzdWNjZXNzO1xuICAgICAgICByZXR1cm4ge3N1Y2Nlc3MsIGFzeW5jSW5pdDogdHJ1ZX07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXSA9IGJhY2tlbmQgYXMgS2VybmVsQmFja2VuZDtcbiAgICAgICAgcmV0dXJuIHtzdWNjZXNzOiB0cnVlLCBhc3luY0luaXQ6IGZhbHNlfTtcbiAgICAgIH1cbiAgICB9IGNhdGNoIChlcnIpIHtcbiAgICAgIGxvZy53YXJuKGBJbml0aWFsaXphdGlvbiBvZiBiYWNrZW5kICR7YmFja2VuZE5hbWV9IGZhaWxlZGApO1xuICAgICAgbG9nLndhcm4oZXJyLnN0YWNrIHx8IGVyci5tZXNzYWdlKTtcbiAgICAgIHJldHVybiB7c3VjY2VzczogZmFsc2UsIGFzeW5jSW5pdDogZmFsc2V9O1xuICAgIH1cbiAgfVxuXG4gIHJlbW92ZUJhY2tlbmQoYmFja2VuZE5hbWU6IHN0cmluZyk6IHZvaWQge1xuICAgIGlmICghKGJhY2tlbmROYW1lIGluIHRoaXMucmVnaXN0cnlGYWN0b3J5KSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGAke2JhY2tlbmROYW1lfSBiYWNrZW5kIG5vdCBmb3VuZCBpbiByZWdpc3RyeWApO1xuICAgIH1cbiAgICBpZiAodGhpcy5iYWNrZW5kTmFtZSA9PT0gYmFja2VuZE5hbWUgJiYgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgIT0gbnVsbCkge1xuICAgICAgLy8gVGhlcmUgaXMgYSBwZW5kaW5nIHByb21pc2Ugb2YgdGhlIGJhY2tlbmQgd2Ugd2FudCB0byByZW1vdmUuIE1ha2UgaXRcbiAgICAgIC8vIG9ic29sZXRlLlxuICAgICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXRJZCsrO1xuICAgIH1cblxuICAgIGlmIChiYWNrZW5kTmFtZSBpbiB0aGlzLnJlZ2lzdHJ5KSB7XG4gICAgICB0aGlzLmRpc3Bvc2VSZWdpc3RlcmVkS2VybmVscyhiYWNrZW5kTmFtZSk7XG4gICAgICB0aGlzLnJlZ2lzdHJ5W2JhY2tlbmROYW1lXS5kaXNwb3NlKCk7XG4gICAgICBkZWxldGUgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV07XG4gICAgfVxuXG4gICAgZGVsZXRlIHRoaXMucmVnaXN0cnlGYWN0b3J5W2JhY2tlbmROYW1lXTtcblxuICAgIC8vIFVuc2V0IHRoZSBiYWNrZW5kIGlmIGl0IGlzIGFjdGl2ZS5cbiAgICBpZiAodGhpcy5iYWNrZW5kTmFtZSA9PT0gYmFja2VuZE5hbWUpIHtcbiAgICAgIHRoaXMucGVuZGluZ0JhY2tlbmRJbml0ID0gbnVsbDtcbiAgICAgIHRoaXMuYmFja2VuZE5hbWUgPSBudWxsO1xuICAgICAgdGhpcy5iYWNrZW5kSW5zdGFuY2UgPSBudWxsO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZ2V0U29ydGVkQmFja2VuZHMoKTogc3RyaW5nW10ge1xuICAgIGlmIChPYmplY3Qua2V5cyh0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkubGVuZ3RoID09PSAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ05vIGJhY2tlbmQgZm91bmQgaW4gcmVnaXN0cnkuJyk7XG4gICAgfVxuICAgIHJldHVybiBPYmplY3Qua2V5cyh0aGlzLnJlZ2lzdHJ5RmFjdG9yeSkuc29ydCgoYTogc3RyaW5nLCBiOiBzdHJpbmcpID0+IHtcbiAgICAgIC8vIEhpZ2hlc3QgcHJpb3JpdHkgY29tZXMgZmlyc3QuXG4gICAgICByZXR1cm4gdGhpcy5yZWdpc3RyeUZhY3RvcnlbYl0ucHJpb3JpdHkgLVxuICAgICAgICAgIHRoaXMucmVnaXN0cnlGYWN0b3J5W2FdLnByaW9yaXR5O1xuICAgIH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBpbml0aWFsaXplQmFja2VuZHNBbmRSZXR1cm5CZXN0KCk6XG4gICAgICB7bmFtZTogc3RyaW5nLCBhc3luY0luaXQ6IGJvb2xlYW59IHtcbiAgICBjb25zdCBzb3J0ZWRCYWNrZW5kcyA9IHRoaXMuZ2V0U29ydGVkQmFja2VuZHMoKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgc29ydGVkQmFja2VuZHMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IGJhY2tlbmROYW1lID0gc29ydGVkQmFja2VuZHNbaV07XG4gICAgICBjb25zdCB7c3VjY2VzcywgYXN5bmNJbml0fSA9IHRoaXMuaW5pdGlhbGl6ZUJhY2tlbmQoYmFja2VuZE5hbWUpO1xuICAgICAgaWYgKGFzeW5jSW5pdCB8fCBzdWNjZXNzKSB7XG4gICAgICAgIHJldHVybiB7bmFtZTogYmFja2VuZE5hbWUsIGFzeW5jSW5pdH07XG4gICAgICB9XG4gICAgfVxuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYENvdWxkIG5vdCBpbml0aWFsaXplIGFueSBiYWNrZW5kcywgYWxsIGJhY2tlbmQgaW5pdGlhbGl6YXRpb25zIGAgK1xuICAgICAgICBgZmFpbGVkLmApO1xuICB9XG5cbiAgbW92ZURhdGEoYmFja2VuZDogS2VybmVsQmFja2VuZCwgZGF0YUlkOiBEYXRhSWQpIHtcbiAgICBjb25zdCBpbmZvID0gdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHNyY0JhY2tlbmQgPSBpbmZvLmJhY2tlbmQ7XG4gICAgY29uc3QgdmFsdWVzID0gdGhpcy5yZWFkU3luYyhkYXRhSWQpO1xuICAgIGNvbnN0IHJlZkNvdW50ID0gc3JjQmFja2VuZC5yZWZDb3VudChkYXRhSWQpO1xuICAgIC8vIERlbGV0ZSB0aGUgdGVuc29yIGZyb20gdGhlIG9sZCBiYWNrZW5kIGFuZCBtb3ZlIGl0IHRvIHRoZSBuZXdcbiAgICAvLyBiYWNrZW5kLlxuICAgIHNyY0JhY2tlbmQuZGlzcG9zZURhdGEoZGF0YUlkLCB0cnVlKTtcbiAgICBpbmZvLmJhY2tlbmQgPSBiYWNrZW5kO1xuICAgIGJhY2tlbmQubW92ZShkYXRhSWQsIHZhbHVlcywgaW5mby5zaGFwZSwgaW5mby5kdHlwZSwgcmVmQ291bnQpO1xuICAgIGlmICh0aGlzLnNob3VsZENoZWNrRm9yTWVtTGVha3MoKSkge1xuICAgICAgLy8gVHJhY2sgdGhlIG51bWJlciBvZiBtb3ZlcyBkdXJpbmcgYSBrZXJuZWwgZXhlY3V0aW9uIHRvIGNvcnJlY3RseVxuICAgICAgLy8gZGV0ZWN0IG1lbW9yeSBsZWFrcy5cbiAgICAgIHRoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2tbdGhpcy5zdGF0ZS5udW1EYXRhTW92ZXNTdGFjay5sZW5ndGggLSAxXSsrO1xuICAgIH1cbiAgfVxuXG4gIHRpZHk8VCBleHRlbmRzIFRlbnNvckNvbnRhaW5lcj4obmFtZU9yRm46IHN0cmluZ3xTY29wZUZuPFQ+LCBmbj86IFNjb3BlRm48VD4pOlxuICAgICAgVCB7XG4gICAgbGV0IG5hbWU6IHN0cmluZyA9IG51bGw7XG4gICAgaWYgKGZuID09IG51bGwpIHtcbiAgICAgIC8vIENhbGxlZCB3aXRoIG9ubHkgMSBhcmd1bWVudC5cbiAgICAgIGlmICh0eXBlb2YgbmFtZU9yRm4gIT09ICdmdW5jdGlvbicpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdQbGVhc2UgcHJvdmlkZSBhIGZ1bmN0aW9uIHRvIHRpZHkoKScpO1xuICAgICAgfVxuICAgICAgZm4gPSBuYW1lT3JGbjtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gQ2FsbGVkIHdpdGggMiBhcmd1bWVudHMuXG4gICAgICBpZiAodHlwZW9mIG5hbWVPckZuICE9PSAnc3RyaW5nJyAmJiAhKG5hbWVPckZuIGluc3RhbmNlb2YgU3RyaW5nKSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnV2hlbiBjYWxsaW5nIHdpdGggdHdvIGFyZ3VtZW50cywgdGhlIGZpcnN0IGFyZ3VtZW50ICcgK1xuICAgICAgICAgICAgJ3RvIHRpZHkoKSBtdXN0IGJlIGEgc3RyaW5nJyk7XG4gICAgICB9XG4gICAgICBpZiAodHlwZW9mIGZuICE9PSAnZnVuY3Rpb24nKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdXaGVuIGNhbGxpbmcgd2l0aCB0d28gYXJndW1lbnRzLCB0aGUgMm5kIGFyZ3VtZW50ICcgK1xuICAgICAgICAgICAgJ3RvIHRpZHkoKSBtdXN0IGJlIGEgZnVuY3Rpb24nKTtcbiAgICAgIH1cbiAgICAgIG5hbWUgPSBuYW1lT3JGbiBhcyBzdHJpbmc7XG4gICAgICAvLyBUT0RPKG5zdGhvcmF0LHNtaWxrb3YpOiBEbyBvcGVyYXRpb24gbG9nZ2luZyBhbmQgcGVyZm9ybWFuY2VcbiAgICAgIC8vIHByb2ZpbGluZy5cbiAgICB9XG4gICAgbGV0IHJlc3VsdDogVDtcbiAgICByZXR1cm4gdGhpcy5zY29wZWRSdW4oXG4gICAgICAgICgpID0+IHRoaXMuc3RhcnRTY29wZShuYW1lKSwgKCkgPT4gdGhpcy5lbmRTY29wZShyZXN1bHQpLCAoKSA9PiB7XG4gICAgICAgICAgcmVzdWx0ID0gZm4oKTtcbiAgICAgICAgICBpZiAocmVzdWx0IGluc3RhbmNlb2YgUHJvbWlzZSkge1xuICAgICAgICAgICAgY29uc29sZS5lcnJvcignQ2Fubm90IHJldHVybiBhIFByb21pc2UgaW5zaWRlIG9mIHRpZHkuJyk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgICAgIH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBzY29wZWRSdW48VD4oc3RhcnQ6ICgpID0+IHZvaWQsIGVuZDogKCkgPT4gdm9pZCwgZjogKCkgPT4gVCk6IFQge1xuICAgIHN0YXJ0KCk7XG4gICAgdHJ5IHtcbiAgICAgIGNvbnN0IHJlcyA9IGYoKTtcbiAgICAgIGVuZCgpO1xuICAgICAgcmV0dXJuIHJlcztcbiAgICB9IGNhdGNoIChleCkge1xuICAgICAgZW5kKCk7XG4gICAgICB0aHJvdyBleDtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIHN0YXRpYyBuZXh0VGVuc29ySWQgPSAwO1xuICBwcml2YXRlIG5leHRUZW5zb3JJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBFbmdpbmUubmV4dFRlbnNvcklkKys7XG4gIH1cblxuICBwcml2YXRlIHN0YXRpYyBuZXh0VmFyaWFibGVJZCA9IDA7XG4gIHByaXZhdGUgbmV4dFZhcmlhYmxlSWQoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gRW5naW5lLm5leHRWYXJpYWJsZUlkKys7XG4gIH1cblxuICAvKipcbiAgICogVGhpcyBtZXRob2QgaXMgY2FsbGVkIGluc3RlYWQgb2YgdGhlIHB1YmxpYy1mYWNpbmcgdGVuc29yLmNsb25lKCkgd2hlblxuICAgKiBzYXZpbmcgYSB0ZW5zb3IgZm9yIGJhY2t3YXJkcyBwYXNzLiBJdCBtYWtlcyBzdXJlIHRvIGFkZCB0aGUgY2xvbmVcbiAgICogb3BlcmF0aW9uIHRvIHRoZSB0YXBlIHJlZ2FyZGxlc3Mgb2YgYmVpbmcgY2FsbGVkIGluc2lkZSBhIGtlcm5lbFxuICAgKiBleGVjdXRpb24uXG4gICAqL1xuICBwcml2YXRlIGNsb25lKHg6IFRlbnNvcik6IFRlbnNvciB7XG4gICAgY29uc3QgeTogVGVuc29yID0gRU5HSU5FLnJ1bktlcm5lbChJZGVudGl0eSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHt4fSBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbiAgICBjb25zdCBpbnB1dHMgPSB7eH07XG4gICAgY29uc3QgZ3JhZCA9IChkeTogVGVuc29yKSA9PiAoe1xuICAgICAgeDogKCkgPT4ge1xuICAgICAgICBjb25zdCBkdHlwZSA9ICdmbG9hdDMyJztcbiAgICAgICAgY29uc3QgZ3JhZElucHV0cyA9IHt4OiBkeX07XG4gICAgICAgIGNvbnN0IGF0dHJzID0ge2R0eXBlfTtcblxuICAgICAgICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgICBDYXN0LCBncmFkSW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICAgICAgICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUZW5zb3I7XG4gICAgICB9XG4gICAgfSk7XG4gICAgY29uc3Qgc2F2ZWQ6IFRlbnNvcltdID0gW107XG4gICAgdGhpcy5hZGRUYXBlTm9kZSh0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLm5hbWUsIGlucHV0cywgW3ldLCBncmFkLCBzYXZlZCwge30pO1xuICAgIHJldHVybiB5O1xuICB9XG5cbiAgLyoqXG4gICAqIEV4ZWN1dGUgYSBrZXJuZWwgd2l0aCB0aGUgZ2l2ZW4gbmFtZSBhbmQgcmV0dXJuIHRoZSBvdXRwdXQgdGVuc29yLlxuICAgKlxuICAgKiBAcGFyYW0ga2VybmVsTmFtZSBUaGUgbmFtZSBvZiB0aGUga2VybmVsIHRvIGV4ZWN1dGUuXG4gICAqIEBwYXJhbSBpbnB1dHMgQSBtYXAgb2YgaW5wdXQgbmFtZXMgdG8gdGVuc29ycy5cbiAgICogQHBhcmFtIGF0dHJzIEEgbWFwIG9mIGF0dHJpYnV0ZSBuYW1lcyB0byB0aGVpciB2YWx1ZXMuIEFuIGF0dHJpYnV0ZSBpcyBhXG4gICAqICAgICBwcmltaXRpdmUgKG5vbi10ZW5zb3IpIGlucHV0IHRvIHRoZSBrZXJuZWwuXG4gICAqIEBwYXJhbSBpbnB1dHNUb1NhdmUgQSBsaXN0IG9mIHRlbnNvcnMsIGlucHV0cyB0byBzYXZlIGZvciB0aGUgYmFja3Byb3BcbiAgICogICAgIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gb3V0cHV0c1RvU2F2ZSBBIGxpc3Qgb2YgYm9vbGVhbnMsIHNwZWNpZnlpbmcgd2hpY2ggb3V0cHV0IHRvIHNhdmVcbiAgICogICAgIGZvciB0aGUgYmFja3Byb3AgY29tcHV0YXRpb24uIFRoZXNlIGFyZSBib29sZWFucyBzaW5jZSB0aGUgb3V0cHV0XG4gICAqIHRlbnNvcnMgYXJlIG5vdCB2aXNpYmxlIHRvIHRoZSB1c2VyLlxuICAgKi9cbiAgcnVuS2VybmVsPFQgZXh0ZW5kcyBUZW5zb3J8VGVuc29yW10+KFxuICAgICAga2VybmVsTmFtZTogc3RyaW5nLCBpbnB1dHM6IE5hbWVkVGVuc29yTWFwLCBhdHRycz86IE5hbWVkQXR0ck1hcCk6IFQge1xuICAgIGlmICh0aGlzLmJhY2tlbmROYW1lID09IG51bGwpIHtcbiAgICAgIC8vIGJhY2tlbmQgaGFzIG5vdCBiZWVuIGluaXRpYWxpemVkIHlldCAoYmFja2VuZCBpbml0aWFsaXphdGlvbiBpcyBsYXp5XG4gICAgICAvLyBjYW4gYmUgZGVmZXJyZWQgdW50aWwgYW4gb3AvIGtlcm5lbCBpcyBydW4pLlxuICAgICAgLy8gVGhlIGJlbG93IGdldHRlciBoYXMgc2lkZSBlZmZlY3RzIHRoYXQgd2lsbCB0cnkgdG8gaW5pdGlhbGl6ZSB0aGVcbiAgICAgIC8vIGJhY2tlbmQgYW5kIHNldCBwcm9wZXJ0aWVzIGxpa2UgdGhpcy5iYWNrZW5kTmFtZVxuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bnVzZWQtZXhwcmVzc2lvblxuICAgICAgdGhpcy5iYWNrZW5kO1xuICAgIH1cbiAgICBjb25zdCBoYXNLZXJuZWwgPSBnZXRLZXJuZWwoa2VybmVsTmFtZSwgdGhpcy5iYWNrZW5kTmFtZSkgIT0gbnVsbDtcbiAgICBpZiAoIWhhc0tlcm5lbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBLZXJuZWwgJyR7a2VybmVsTmFtZX0nIG5vdCByZWdpc3RlcmVkIGZvciBiYWNrZW5kICcke1xuICAgICAgICAgIHRoaXMuYmFja2VuZE5hbWV9J2ApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5ydW5LZXJuZWxGdW5jKHtrZXJuZWxOYW1lLCBpbnB1dHMsIGF0dHJzfSk7XG4gIH1cblxuICBwcml2YXRlIHNob3VsZENoZWNrRm9yTWVtTGVha3MoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuRU5WLmdldEJvb2woJ0lTX1RFU1QnKTtcbiAgfVxuXG4gIHByaXZhdGUgY2hlY2tLZXJuZWxGb3JNZW1MZWFrKFxuICAgICAga2VybmVsTmFtZTogc3RyaW5nLCBudW1EYXRhSWRzQmVmb3JlOiBudW1iZXIsXG4gICAgICBvdXRJbmZvczogVGVuc29ySW5mb1tdKTogdm9pZCB7XG4gICAgY29uc3QgbnVtRGF0YUlkc0FmdGVyID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcblxuICAgIC8vIENvdW50IHRoZSBudW1iZXIgb2YgZGF0YSBpZHMgYXNzb2NpYXRlZCB3aXRoIHRoZSByZXN1bHQgb2YgdGhlIGtlcm5lbC5cbiAgICBsZXQgbnVtT3V0cHV0RGF0YUlkcyA9IDA7XG4gICAgb3V0SW5mb3MuZm9yRWFjaChpbmZvID0+IHtcbiAgICAgIC8vIENvbXBsZXggbnVtYmVycyBhbGxvY2F0ZSAzIGRhdGEgaWRzLCBvbmUgZm9yICdyZWFsJywgb25lIGZvclxuICAgICAgLy8gJ2ltYWdpbmFyeScsIGFuZCBvbmUgZm9yIHRoZSBjb250YWluZXIgdGhhdCBob2xkcyB0aGUgZm9ybWVyIHR3by5cbiAgICAgIG51bU91dHB1dERhdGFJZHMgKz0gKGluZm8uZHR5cGUgPT09ICdjb21wbGV4NjQnID8gMyA6IDEpO1xuICAgIH0pO1xuXG4gICAgLy8gQWNjb3VudCBmb3IgdGhlIG51bWJlciBvZiBtb3ZlcyBkdXJpbmcga2VybmVsIGV4ZWN1dGlvbi4gQSBcImRhdGEgbW92ZVwiXG4gICAgLy8gY2FuIGhhcHBlbiBpbiB0aGUgbWlkZGxlIG9mIGEga2VybmVsIGV4ZWN1dGlvbiwgcGxhY2luZyBhIG5ldyAoa2V5LHZhbHVlKVxuICAgIC8vIHBhaXIgaW4gdGhlIGRhdGEgc3RvcmFnZS4gU2luY2UgZGF0YSBtb3ZlcyBoYXZlIG5ldCB6ZXJvIGVmZmVjdCAod2VcbiAgICAvLyBhbHdheXMgcmVtb3ZlIHRoZSBkYXRhIGZyb20gdGhlIG9sZCBiYWNrZW5kKSwgd2UgaGF2ZSB0byBjYW5jZWwgdGhlbSBvdXRcbiAgICAvLyB3aGVuIGRldGVjdGluZyBtZW1vcnkgbGVha3MuXG4gICAgY29uc3QgbnVtTW92ZXMgPVxuICAgICAgICB0aGlzLnN0YXRlLm51bURhdGFNb3Zlc1N0YWNrW3RoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2subGVuZ3RoIC0gMV07XG4gICAgY29uc3QgZGF0YUlkc0xlYWtlZCA9XG4gICAgICAgIG51bURhdGFJZHNBZnRlciAtIG51bURhdGFJZHNCZWZvcmUgLSBudW1PdXRwdXREYXRhSWRzIC0gbnVtTW92ZXM7XG4gICAgaWYgKGRhdGFJZHNMZWFrZWQgPiAwKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYEJhY2tlbmQgJyR7dGhpcy5iYWNrZW5kTmFtZX0nIGhhcyBhbiBpbnRlcm5hbCBtZW1vcnkgbGVhayBgICtcbiAgICAgICAgICBgKCR7ZGF0YUlkc0xlYWtlZH0gZGF0YSBpZHMpIGFmdGVyIHJ1bm5pbmcgJyR7a2VybmVsTmFtZX0nYCk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIEludGVybmFsIGhlbHBlciBtZXRob2QgdG8gZXhlY3V0ZSBhIGtlcm5lbCBGdW5jXG4gICAqXG4gICAqIFVzZSBgcnVuS2VybmVsYCB0byBleGVjdXRlIGtlcm5lbHMgZnJvbSBvdXRzaWRlIG9mIGVuZ2luZS5cbiAgICovXG4gIHByaXZhdGUgcnVuS2VybmVsRnVuYzxUIGV4dGVuZHMgVGVuc29yfFRlbnNvcltdLCBJIGV4dGVuZHMgTmFtZWRUZW5zb3JNYXA+KFxuICAgICAga2VybmVsUGFyYW1zOiBSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbjxJPnxcbiAgICAgIEN1c3RvbUdyYWRLZXJuZWxJbnZvY2F0aW9uPFQsIEk+KTogVCB7XG4gICAgbGV0IG91dHB1dHM6IFRlbnNvcltdO1xuICAgIGxldCBzYXZlZDogVGVuc29yW10gPSBbXTtcbiAgICBjb25zdCBpc1RhcGVPbiA9IHRoaXMuaXNUYXBlT24oKTtcblxuICAgIGNvbnN0IHN0YXJ0aW5nQnl0ZWNvdW50ID0gdGhpcy5zdGF0ZS5udW1CeXRlcztcbiAgICBjb25zdCBzdGFydGluZ051bVRlbnNvcnMgPSB0aGlzLnN0YXRlLm51bVRlbnNvcnM7XG5cbiAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgIHRoaXMuc3RhdGUubnVtRGF0YU1vdmVzU3RhY2sucHVzaCgwKTtcbiAgICB9XG5cbiAgICBsZXQga2VybmVsRnVuYzogKCkgPT4gVGVuc29yW107XG4gICAgaWYgKHRoaXMuYmFja2VuZE5hbWUgPT0gbnVsbCkge1xuICAgICAgLy8gYmFja2VuZCBoYXMgbm90IGJlZW4gaW5pdGlhbGl6ZWQgeWV0IChiYWNrZW5kIGluaXRpYWxpemF0aW9uIGlzIGxhenlcbiAgICAgIC8vIGNhbiBiZSBkZWZlcnJlZCB1bnRpbCBhbiBvcC8ga2VybmVsIGlzIHJ1bikuXG4gICAgICAvLyBUaGUgYmVsb3cgZ2V0dGVyIGhhcyBzaWRlIGVmZmVjdHMgdGhhdCB3aWxsIHRyeSB0byBpbml0aWFsaXplIHRoZVxuICAgICAgLy8gYmFja2VuZCBhbmQgc2V0IHByb3BlcnRpZXMgbGlrZSB0aGlzLmJhY2tlbmROYW1lXG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgICB0aGlzLmJhY2tlbmQ7XG4gICAgfVxuXG4gICAgbGV0IG91dDogVGVuc29ySW5mb3xUZW5zb3JJbmZvW107XG5cbiAgICBjb25zdCBrZXJuZWxPclNjb3BlTmFtZSA9IGlzUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb24oa2VybmVsUGFyYW1zKSA/XG4gICAgICAgIGtlcm5lbFBhcmFtcy5rZXJuZWxOYW1lIDpcbiAgICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZSAhPSBudWxsID8gdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS5uYW1lIDogJyc7XG5cbiAgICAvLyBDcmVhdGUgdGhlIGtlcm5lbEZ1bmMgZnJvbSBlaXRoZXIgYSByZWdpc3RlcmVkIGtlcm5lbCBPUiBwYXNzZWQgaW5cbiAgICAvLyBmb3J3YXJkL2JhY2t3YXJkIGZ1bmN0aW9ucyAodXNlZCBieSBjdXN0b20gZ3JhZCkuIEluIHRoaXMgY29udGV4dCBhXG4gICAgLy8ga2VybmVsRnVuYyB3cmFwcyBhIGtlcm5lbCBpbXBsZW1lbnRhdGlvbiB3aXRoIHNvbWUgYm9va2tlZXBpbmcuXG5cbiAgICBpZiAoaXNSZWdpc3RlcmVkS2VybmVsSW52b2NhdGlvbihrZXJuZWxQYXJhbXMpKSB7XG4gICAgICBjb25zdCB7a2VybmVsTmFtZSwgaW5wdXRzLCBhdHRyc30gPSBrZXJuZWxQYXJhbXM7XG4gICAgICBpZiAodGhpcy5iYWNrZW5kTmFtZSA9PSBudWxsKSB7XG4gICAgICAgIC8vIGJhY2tlbmQgaGFzIG5vdCBiZWVuIGluaXRpYWxpemVkIHlldCAoYmFja2VuZCBpbml0aWFsaXphdGlvbiBpcyBsYXp5XG4gICAgICAgIC8vIGNhbiBiZSBkZWZlcnJlZCB1bnRpbCBhbiBvcC8ga2VybmVsIGlzIHJ1bikuXG4gICAgICAgIC8vIFRoZSBiZWxvdyBnZXR0ZXIgaGFzIHNpZGUgZWZmZWN0cyB0aGF0IHdpbGwgdHJ5IHRvIGluaXRpYWxpemUgdGhlXG4gICAgICAgIC8vIGJhY2tlbmQgYW5kIHNldCBwcm9wZXJ0aWVzIGxpa2UgdGhpcy5iYWNrZW5kTmFtZVxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVudXNlZC1leHByZXNzaW9uXG4gICAgICAgIHRoaXMuYmFja2VuZDtcbiAgICAgIH1cbiAgICAgIGNvbnN0IGtlcm5lbCA9IGdldEtlcm5lbChrZXJuZWxOYW1lLCB0aGlzLmJhY2tlbmROYW1lKTtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGtlcm5lbCAhPSBudWxsLFxuICAgICAgICAgICgpID0+IGBDYW5ub3QgZmluZCByZWdpc3RlcmVkIGtlcm5lbCAnJHtrZXJuZWxOYW1lfScgZm9yIGJhY2tlbmQgJyR7XG4gICAgICAgICAgICAgIHRoaXMuYmFja2VuZE5hbWV9J2ApO1xuXG4gICAgICBrZXJuZWxGdW5jID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBudW1EYXRhSWRzQmVmb3JlID0gdGhpcy5iYWNrZW5kLm51bURhdGFJZHMoKTtcbiAgICAgICAgb3V0ID0ga2VybmVsLmtlcm5lbEZ1bmMoe2lucHV0cywgYXR0cnMsIGJhY2tlbmQ6IHRoaXMuYmFja2VuZH0pO1xuICAgICAgICBjb25zdCBvdXRJbmZvcyA9IEFycmF5LmlzQXJyYXkob3V0KSA/IG91dCA6IFtvdXRdO1xuICAgICAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgICAgICB0aGlzLmNoZWNrS2VybmVsRm9yTWVtTGVhayhrZXJuZWxOYW1lLCBudW1EYXRhSWRzQmVmb3JlLCBvdXRJbmZvcyk7XG4gICAgICAgIH1cblxuICAgICAgICBjb25zdCBvdXRUZW5zb3JzID0gb3V0SW5mb3MubWFwKChvdXRJbmZvOiBUZW5zb3JJbmZvfFRlbnNvcikgPT4ge1xuICAgICAgICAgIC8vIHRvZG8gKHlhc3NvZ2JhKSByZW1vdmUgdGhpcyBvcHRpb24gKFRlbnNvcikgd2hlbiBub2RlIGJhY2tlbmRcbiAgICAgICAgICAvLyBtZXRob2RzIGhhdmUgYmVlbiBtb2R1bGFyaXplZCBhbmQgdGhleSBhbGwgcmV0dXJuIHRlbnNvckluZm8uXG4gICAgICAgICAgLy8gVGVuc29ySW5mb3MgZG8gbm90IGhhdmUgYSByYW5rIGF0dHJpYnV0ZS5cbiAgICAgICAgICBpZiAoKG91dEluZm8gYXMgVGVuc29yKS5yYW5rICE9IG51bGwpIHtcbiAgICAgICAgICAgIHJldHVybiBvdXRJbmZvIGFzIFRlbnNvcjtcbiAgICAgICAgICB9XG4gICAgICAgICAgcmV0dXJuIHRoaXMubWFrZVRlbnNvckZyb21UZW5zb3JJbmZvKG91dEluZm8pO1xuICAgICAgICB9KTtcblxuICAgICAgICAvLyBTYXZlIGFueSByZXF1aXJlZCBpbnB1dHMgYW5kIG91dHB1dHMuXG5cbiAgICAgICAgLy8gRG8gbm90IHNhdmUgdW5sZXNzIHdlIGFyZSByZWNvcmRpbmcgdG8gdGhlIHRhcGUuIE90aGVyd2lzZSBpdCB3b3VsZFxuICAgICAgICAvLyBjYXVzZSBhIG1lbSBsZWFrIHNpbmNlIHRoZXJlIHdvdWxkIGJlIG5vIGJhY2twcm9wIGZvciB0aGVzZSB0ZW5zb3JzXG4gICAgICAgIC8vICh3aGljaCB3b3VsZCBvdGhlcndpc2UgZGlzcG9zZSB0aGVtKS5cbiAgICAgICAgaWYgKGlzVGFwZU9uKSB7XG4gICAgICAgICAgY29uc3QgdGVuc29yc1RvU2F2ZSA9XG4gICAgICAgICAgICAgIHRoaXMuZ2V0VGVuc29yc0ZvckdyYWRpZW50KGtlcm5lbE5hbWUsIGlucHV0cywgb3V0VGVuc29ycyk7XG4gICAgICAgICAgc2F2ZWQgPSB0aGlzLnNhdmVUZW5zb3JzRm9yQmFja3dhcmRNb2RlKHRlbnNvcnNUb1NhdmUpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBvdXRUZW5zb3JzO1xuICAgICAgfTtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3Qge2ZvcndhcmRGdW5jfSA9IGtlcm5lbFBhcmFtcztcbiAgICAgIC8vIFJ1bm5pbmcgYSBjdXN0b21HcmFkIG9wLlxuICAgICAgY29uc3Qgc2F2ZUZ1bmM6IEdyYWRTYXZlRnVuYyA9ICh0ZW5zb3JzKSA9PiB7XG4gICAgICAgIC8vIERvIG5vdCBzYXZlIHVubGVzcyB3ZSBhcmUgcmVjb3JkaW5nIHRvIHRoZSB0YXBlLiBPdGhlcndpc2UgaXQgd291bGRcbiAgICAgICAgLy8gY2F1c2UgYSBtZW0gbGVhayBzaW5jZSB3ZSB3b3VsZCBuZXZlciBydW4gYmFja3Byb3AsIHdoaWNoIGRpc3Bvc2VzXG4gICAgICAgIC8vIHRoZSBrZXB0IHRlbnNvcnMuXG4gICAgICAgIGlmICghaXNUYXBlT24pIHtcbiAgICAgICAgICByZXR1cm47XG4gICAgICAgIH1cbiAgICAgICAgc2F2ZWQgPSB0ZW5zb3JzLm1hcCh0ZW5zb3IgPT4gdGhpcy5rZWVwKHRoaXMuY2xvbmUodGVuc29yKSkpO1xuICAgICAgfTtcblxuICAgICAga2VybmVsRnVuYyA9ICgpID0+IHtcbiAgICAgICAgY29uc3QgbnVtRGF0YUlkc0JlZm9yZSA9IHRoaXMuYmFja2VuZC5udW1EYXRhSWRzKCk7XG4gICAgICAgIG91dCA9IHRoaXMudGlkeSgoKSA9PiBmb3J3YXJkRnVuYyh0aGlzLmJhY2tlbmQsIHNhdmVGdW5jKSk7XG4gICAgICAgIGNvbnN0IG91dHMgPSAoQXJyYXkuaXNBcnJheShvdXQpID8gb3V0IDogW291dF0pIGFzIFRlbnNvcltdO1xuICAgICAgICBpZiAodGhpcy5zaG91bGRDaGVja0Zvck1lbUxlYWtzKCkpIHtcbiAgICAgICAgICAvLyBTY29wZSBuYW1lIGlzIHVzZWQgdG8gcHJpbnQgYSBtb3JlIGhlbHBmdWwgZXJyb3IgbWVzc2FnZSBpZiBuZWVkZWQuXG4gICAgICAgICAgdGhpcy5jaGVja0tlcm5lbEZvck1lbUxlYWsoa2VybmVsT3JTY29wZU5hbWUsIG51bURhdGFJZHNCZWZvcmUsIG91dHMpO1xuICAgICAgICB9XG4gICAgICAgIHJldHVybiBvdXRzO1xuICAgICAgfTtcbiAgICB9XG5cbiAgICAvL1xuICAgIC8vIFJ1biB0aGUga2VybmVsRnVuYy4gT3B0aW9uYWxseSBwcm9maWxpbmcgaXQuXG4gICAgLy9cbiAgICBjb25zdCB7aW5wdXRzLCBhdHRyc30gPSBrZXJuZWxQYXJhbXM7XG4gICAgY29uc3QgYmFja3dhcmRzRnVuYyA9IGlzUmVnaXN0ZXJlZEtlcm5lbEludm9jYXRpb24oa2VybmVsUGFyYW1zKSA/XG4gICAgICAgIG51bGwgOlxuICAgICAgICBrZXJuZWxQYXJhbXMuYmFja3dhcmRzRnVuYztcblxuICAgIGxldCBrZXJuZWxQcm9maWxlOiBLZXJuZWxQcm9maWxlO1xuICAgIHRoaXMuc2NvcGVkUnVuKFxuICAgICAgICAvLyBTdG9wIHJlY29yZGluZyB0byBhIHRhcGUgd2hlbiBydW5uaW5nIGEga2VybmVsLlxuICAgICAgICAoKSA9PiB0aGlzLnN0YXRlLmtlcm5lbERlcHRoKyssICgpID0+IHRoaXMuc3RhdGUua2VybmVsRGVwdGgtLSwgKCkgPT4ge1xuICAgICAgICAgIGlmICghdGhpcy5FTlYuZ2V0Qm9vbCgnREVCVUcnKSAmJiAhdGhpcy5zdGF0ZS5wcm9maWxpbmcpIHtcbiAgICAgICAgICAgIG91dHB1dHMgPSBrZXJuZWxGdW5jKCk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGtlcm5lbFByb2ZpbGUgPSB0aGlzLnByb2ZpbGVyLnByb2ZpbGVLZXJuZWwoXG4gICAgICAgICAgICAgICAga2VybmVsT3JTY29wZU5hbWUsIGlucHV0cywgKCkgPT4ga2VybmVsRnVuYygpKTtcbiAgICAgICAgICAgIGlmICh0aGlzLkVOVi5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgICAgICAgICAgIHRoaXMucHJvZmlsZXIubG9nS2VybmVsUHJvZmlsZShrZXJuZWxQcm9maWxlKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIG91dHB1dHMgPSBrZXJuZWxQcm9maWxlLm91dHB1dHM7XG4gICAgICAgICAgfVxuICAgICAgICB9KTtcblxuICAgIGlmIChpc1RhcGVPbikge1xuICAgICAgdGhpcy5hZGRUYXBlTm9kZShcbiAgICAgICAgICBrZXJuZWxPclNjb3BlTmFtZSwgaW5wdXRzLCBvdXRwdXRzLCBiYWNrd2FyZHNGdW5jLCBzYXZlZCwgYXR0cnMpO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnN0YXRlLnByb2ZpbGluZykge1xuICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLmtlcm5lbHMucHVzaCh7XG4gICAgICAgIG5hbWU6IGtlcm5lbE9yU2NvcGVOYW1lLFxuICAgICAgICBieXRlc0FkZGVkOiB0aGlzLnN0YXRlLm51bUJ5dGVzIC0gc3RhcnRpbmdCeXRlY291bnQsXG4gICAgICAgIHRvdGFsQnl0ZXNTbmFwc2hvdDogdGhpcy5zdGF0ZS5udW1CeXRlcyxcbiAgICAgICAgdGVuc29yc0FkZGVkOiB0aGlzLnN0YXRlLm51bVRlbnNvcnMgLSBzdGFydGluZ051bVRlbnNvcnMsXG4gICAgICAgIHRvdGFsVGVuc29yc1NuYXBzaG90OiB0aGlzLnN0YXRlLm51bVRlbnNvcnMsXG4gICAgICAgIGlucHV0U2hhcGVzOiBPYmplY3Qua2V5cyhpbnB1dHMpLm1hcChcbiAgICAgICAgICAgIGtleSA9PiBpbnB1dHNba2V5XSAhPSBudWxsID8gaW5wdXRzW2tleV0uc2hhcGUgOiBudWxsKSxcbiAgICAgICAgb3V0cHV0U2hhcGVzOiBvdXRwdXRzLm1hcChpdGVtID0+IGl0ZW0uc2hhcGUpLFxuICAgICAgICBrZXJuZWxUaW1lTXM6IGtlcm5lbFByb2ZpbGUudGltZU1zLFxuICAgICAgICBleHRyYUluZm86IGtlcm5lbFByb2ZpbGUuZXh0cmFJbmZvXG4gICAgICB9KTtcbiAgICB9XG4gICAgcmV0dXJuIChBcnJheS5pc0FycmF5KG91dCkgPyBvdXRwdXRzIDogb3V0cHV0c1swXSkgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBTYXZlcyB0ZW5zb3JzIHVzZWQgaW4gZm9yd2FyZCBtb2RlIGZvciB1c2UgaW4gYmFja3dhcmQgbW9kZS5cbiAgICpcbiAgICogQHBhcmFtIHRlbnNvcnMgdGhlIGxpc3Qgb2YgdGVuc29ycyB0byBzYXZlLlxuICAgKi9cbiAgcHJpdmF0ZSBzYXZlVGVuc29yc0ZvckJhY2t3YXJkTW9kZSh0ZW5zb3JzOiBUZW5zb3JbXSk6IFRlbnNvcltdIHtcbiAgICBjb25zdCBzYXZlZCA9IHRlbnNvcnMubWFwKHRlbnNvciA9PiB0aGlzLmtlZXAodGhpcy5jbG9uZSh0ZW5zb3IpKSk7XG4gICAgcmV0dXJuIHNhdmVkO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSBsaXN0IG9mIHRlbnNvcnMgdG8gc2F2ZSBmb3IgYSBnaXZlbiBncmFkaWVudCBjYWxjdWxhdGlvbi5cbiAgICpcbiAgICogQHBhcmFtIGtlcm5lbE5hbWUgbmFtZSBvZiBrZXJuZWwgdG8gbG9vayB1cCBncmFkaWVudCBmb3IuXG4gICAqIEBwYXJhbSBpbnB1dHMgYSBtYXAgb2YgaW5wdXQgdGVuc29ycy5cbiAgICogQHBhcmFtIG91dHB1dHMgYW4gYXJyYXkgb2Ygb3V0cHV0IHRlbnNvcnMgZnJvbSBmb3J3YXJkIG1vZGUgb2Yga2VybmVsLlxuICAgKi9cbiAgcHJpdmF0ZSBnZXRUZW5zb3JzRm9yR3JhZGllbnQoXG4gICAgICBrZXJuZWxOYW1lOiBzdHJpbmcsIGlucHV0czogTmFtZWRUZW5zb3JNYXAsXG4gICAgICBvdXRwdXRzOiBUZW5zb3JbXSk6IFRlbnNvcltdfG51bGwge1xuICAgIGNvbnN0IGdyYWRDb25maWcgPSBnZXRHcmFkaWVudChrZXJuZWxOYW1lKTtcbiAgICBpZiAoZ3JhZENvbmZpZyAhPSBudWxsKSB7XG4gICAgICBjb25zdCBpbnB1dHNUb1NhdmU6IHN0cmluZ1tdID0gZ3JhZENvbmZpZy5pbnB1dHNUb1NhdmUgfHwgW107XG4gICAgICBjb25zdCBvdXRwdXRzVG9TYXZlOiBib29sZWFuW10gPSBncmFkQ29uZmlnLm91dHB1dHNUb1NhdmUgfHwgW107XG5cbiAgICAgIC8vIElmIHNhdmVBbGxJbnB1dHMgaXMgdHJ1ZSwgYWxsIGlucHV0cyB3aWxsIGJlIHNhdmVkLiBPdGhlcndpc2UsIGlucHV0c1xuICAgICAgLy8gc3BlY2lmaWVkIGluIGlucHV0c1RvU2F2ZSB3aWxsIGJlIHNhdmVkLlxuICAgICAgbGV0IGlucHV0VGVuc29yc1RvU2F2ZTogVGVuc29yW107XG4gICAgICBpZiAoZ3JhZENvbmZpZy5zYXZlQWxsSW5wdXRzKSB7XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgQXJyYXkuaXNBcnJheShpbnB1dHMpLFxuICAgICAgICAgICAgKCkgPT4gJ3NhdmVBbGxJbnB1dHMgaXMgdHJ1ZSwgZXhwZWN0ZWQgaW5wdXRzIHRvIGJlIGFuIGFycmF5LicpO1xuXG4gICAgICAgIGlucHV0VGVuc29yc1RvU2F2ZSA9IE9iamVjdC5rZXlzKGlucHV0cykubWFwKChrZXkpID0+IGlucHV0c1trZXldKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGlucHV0VGVuc29yc1RvU2F2ZSA9IGlucHV0c1RvU2F2ZS5tYXAoKGlucHV0TmFtZSkgPT4gaW5wdXRzW2lucHV0TmFtZV0pO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBvdXRwdXRUZW5zb3JzVG9TYXZlOiBUZW5zb3JbXSA9XG4gICAgICAgICAgb3V0cHV0cy5maWx0ZXIoKF8sIGkpID0+IG91dHB1dHNUb1NhdmVbaV0pO1xuXG4gICAgICByZXR1cm4gaW5wdXRUZW5zb3JzVG9TYXZlLmNvbmNhdChvdXRwdXRUZW5zb3JzVG9TYXZlKTtcbiAgICB9XG4gICAgLy8gV2UgcmV0dXJuIGFuIGVtcHR5IGxpc3QgcmF0aGVyIHRoYW4gdGhyb3cgYW4gZXJyb3IgYmVjYXVzZSB0aGUga2VybmVsIHdlXG4gICAgLy8gYXJlIGxvb2tpbmcgdXAgbWF5IG5vdCBhY3R1YWxseSBiZSByZWxldmFudCB0byBiYWNrcHJvcGluZyB0aHJvdWdoIHRoZVxuICAgIC8vIG92ZXJhbGwgZnVuY3Rpb25cbiAgICAvL1xuICAgIC8vIFNlZSAnZG9lcyBub3QgZXJyb3IgaWYgaXJyZWxldmFudCAocHJ1bmVkKSBvcHMgYXJlIG1pc3NpbmcgZ3JhZHMnIHRlc3RcbiAgICAvLyBpbiBncmFkaWVudHNfdGVzdC50cyBmb3IgYW4gZXhhbXBsZS5cbiAgICByZXR1cm4gW107XG4gIH1cblxuICAvKipcbiAgICogSW50ZXJuYWwgbWV0aG9kIHVzZWQgYnkgcHVibGljIEFQSXMgZm9yIHRlbnNvciBjcmVhdGlvbi4gTWFrZXMgYSBuZXdcbiAgICogdGVuc29yIHdpdGggdGhlIHByb3ZpZGVkIHNoYXBlLCBkdHlwZSBhbmQgdmFsdWVzLiBJdCBhbHdheXNcbiAgICogY3JlYXRlcyBhIG5ldyBkYXRhIGlkIGFuZCB3cml0ZXMgdGhlIHZhbHVlcyB0byB0aGUgdW5kZXJseWluZyBiYWNrZW5kLlxuICAgKi9cbiAgbWFrZVRlbnNvcihcbiAgICAgIHZhbHVlczogRGF0YVZhbHVlcywgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICBiYWNrZW5kPzogS2VybmVsQmFja2VuZCk6IFRlbnNvciB7XG4gICAgaWYgKHZhbHVlcyA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1ZhbHVlcyBwYXNzZWQgdG8gZW5naW5lLm1ha2VUZW5zb3IoKSBhcmUgbnVsbCcpO1xuICAgIH1cbiAgICBkdHlwZSA9IGR0eXBlIHx8ICdmbG9hdDMyJztcbiAgICBiYWNrZW5kID0gYmFja2VuZCB8fCB0aGlzLmJhY2tlbmQ7XG4gICAgbGV0IGJhY2tlbmRWYWxzID0gdmFsdWVzIGFzIEJhY2tlbmRWYWx1ZXM7XG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJyAmJiB1dGlsLmlzU3RyaW5nKHZhbHVlc1swXSkpIHtcbiAgICAgIGJhY2tlbmRWYWxzID0gKHZhbHVlcyBhcyBzdHJpbmdbXSkubWFwKGQgPT4gdXRpbC5lbmNvZGVTdHJpbmcoZCkpO1xuICAgIH1cbiAgICBjb25zdCBkYXRhSWQgPSBiYWNrZW5kLndyaXRlKGJhY2tlbmRWYWxzLCBzaGFwZSwgZHR5cGUpO1xuICAgIGNvbnN0IHQgPSBuZXcgVGVuc29yKHNoYXBlLCBkdHlwZSwgZGF0YUlkLCB0aGlzLm5leHRUZW5zb3JJZCgpKTtcbiAgICB0aGlzLnRyYWNrVGVuc29yKHQsIGJhY2tlbmQpO1xuXG4gICAgLy8gQ291bnQgYnl0ZXMgZm9yIHN0cmluZyB0ZW5zb3JzLlxuICAgIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCk7XG4gICAgICBjb25zdCBuZXdCeXRlcyA9IGJ5dGVzRnJvbVN0cmluZ0FycmF5KGJhY2tlbmRWYWxzIGFzIFVpbnQ4QXJyYXlbXSk7XG4gICAgICB0aGlzLnN0YXRlLm51bUJ5dGVzICs9IG5ld0J5dGVzIC0gaW5mby5ieXRlcztcbiAgICAgIGluZm8uYnl0ZXMgPSBuZXdCeXRlcztcbiAgICB9XG4gICAgcmV0dXJuIHQ7XG4gIH1cblxuICAvKipcbiAgICogSW50ZXJuYWwgbWV0aG9kIHVzZWQgYnkgYmFja2VuZHMuIE1ha2VzIGEgbmV3IHRlbnNvclxuICAgKiB0aGF0IGlzIGEgd3JhcHBlciBhcm91bmQgYW4gZXhpc3RpbmcgZGF0YSBpZC4gSXQgZG9lc24ndCBjcmVhdGVcbiAgICogYSBuZXcgZGF0YSBpZCwgb25seSBpbmNyZW1lbnRzIHRoZSByZWYgY291bnQgdXNlZCBpbiBtZW1vcnkgdHJhY2tpbmcuXG4gICAqIEBkZXByZWNhdGVkXG4gICAqL1xuICBtYWtlVGVuc29yRnJvbURhdGFJZChcbiAgICBkYXRhSWQ6IERhdGFJZCwgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgYmFja2VuZD86IEtlcm5lbEJhY2tlbmQpOiBUZW5zb3Ige1xuICAgIGR0eXBlID0gZHR5cGUgfHwgJ2Zsb2F0MzInO1xuICAgIGNvbnN0IHRlbnNvckluZm86IFRlbnNvckluZm8gPSB7ZGF0YUlkLCBzaGFwZSwgZHR5cGV9O1xuICAgIHJldHVybiB0aGlzLm1ha2VUZW5zb3JGcm9tVGVuc29ySW5mbyh0ZW5zb3JJbmZvLCBiYWNrZW5kKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBJbnRlcm5hbCBtZXRob2QgdXNlZCBieSBiYWNrZW5kcy4gTWFrZXMgYSBuZXcgdGVuc29yIHRoYXQgaXMgYSB3cmFwcGVyXG4gICAqIGFyb3VuZCBhbiBleGlzdGluZyBkYXRhIGlkIGluIFRlbnNvckluZm8uIEl0IGRvZXNuJ3QgY3JlYXRlIGEgbmV3IGRhdGEgaWQsXG4gICAqIG9ubHkgaW5jcmVtZW50cyB0aGUgcmVmIGNvdW50IHVzZWQgaW4gbWVtb3J5IHRyYWNraW5nLlxuICAgKi9cbiAgbWFrZVRlbnNvckZyb21UZW5zb3JJbmZvKHRlbnNvckluZm86IFRlbnNvckluZm8sIGJhY2tlbmQ/OiBLZXJuZWxCYWNrZW5kKTpcbiAgICAgIFRlbnNvciB7XG4gICAgY29uc3Qge2RhdGFJZCwgc2hhcGUsIGR0eXBlfSA9IHRlbnNvckluZm87XG4gICAgY29uc3QgdCA9IG5ldyBUZW5zb3Ioc2hhcGUsIGR0eXBlLCBkYXRhSWQsIHRoaXMubmV4dFRlbnNvcklkKCkpO1xuICAgIHRoaXMudHJhY2tUZW5zb3IodCwgYmFja2VuZCk7XG4gICAgcmV0dXJuIHQ7XG4gIH1cblxuICBtYWtlVmFyaWFibGUoXG4gICAgICBpbml0aWFsVmFsdWU6IFRlbnNvciwgdHJhaW5hYmxlID0gdHJ1ZSwgbmFtZT86IHN0cmluZyxcbiAgICAgIGR0eXBlPzogRGF0YVR5cGUpOiBWYXJpYWJsZSB7XG4gICAgbmFtZSA9IG5hbWUgfHwgdGhpcy5uZXh0VmFyaWFibGVJZCgpLnRvU3RyaW5nKCk7XG4gICAgaWYgKGR0eXBlICE9IG51bGwgJiYgZHR5cGUgIT09IGluaXRpYWxWYWx1ZS5kdHlwZSkge1xuICAgICAgaW5pdGlhbFZhbHVlID0gaW5pdGlhbFZhbHVlLmNhc3QoZHR5cGUpO1xuICAgIH1cbiAgICBjb25zdCB2ID0gbmV3IFZhcmlhYmxlKGluaXRpYWxWYWx1ZSwgdHJhaW5hYmxlLCBuYW1lLCB0aGlzLm5leHRUZW5zb3JJZCgpKTtcbiAgICBpZiAodGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzW3YubmFtZV0gIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBWYXJpYWJsZSB3aXRoIG5hbWUgJHt2Lm5hbWV9IHdhcyBhbHJlYWR5IHJlZ2lzdGVyZWRgKTtcbiAgICB9XG4gICAgdGhpcy5zdGF0ZS5yZWdpc3RlcmVkVmFyaWFibGVzW3YubmFtZV0gPSB2O1xuICAgIHRoaXMuaW5jUmVmKHYsIHRoaXMuYmFja2VuZCk7XG4gICAgcmV0dXJuIHY7XG4gIH1cblxuICB0cmFja1RlbnNvcihhOiBUZW5zb3IsIGJhY2tlbmQ6IEtlcm5lbEJhY2tlbmQpOiB2b2lkIHtcbiAgICB0aGlzLnN0YXRlLm51bVRlbnNvcnMrKztcbiAgICBpZiAoYS5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRoaXMuc3RhdGUubnVtU3RyaW5nVGVuc29ycysrO1xuICAgIH1cbiAgICAvLyBCeXRlcyBmb3IgY29tcGxleCBudW1iZXJzIGFyZSBjb3VudGVkIGJ5IHRoZWlyIGNvbXBvbmVudHMuIEJ5dGVzIGZvclxuICAgIC8vIHN0cmluZyB0ZW5zb3JzIGFyZSBjb3VudGVkIHdoZW4gd3JpdGluZyB2YWx1ZXMuXG4gICAgbGV0IGJ5dGVzID0gMDtcbiAgICBpZiAoYS5kdHlwZSAhPT0gJ2NvbXBsZXg2NCcgJiYgYS5kdHlwZSAhPT0gJ3N0cmluZycpIHtcbiAgICAgIGJ5dGVzID0gYS5zaXplICogdXRpbC5ieXRlc1BlckVsZW1lbnQoYS5kdHlwZSk7XG4gICAgfVxuICAgIHRoaXMuc3RhdGUubnVtQnl0ZXMgKz0gYnl0ZXM7XG5cbiAgICBpZiAoIXRoaXMuc3RhdGUudGVuc29ySW5mby5oYXMoYS5kYXRhSWQpKSB7XG4gICAgICB0aGlzLnN0YXRlLm51bURhdGFCdWZmZXJzKys7XG4gICAgICB0aGlzLnN0YXRlLnRlbnNvckluZm8uc2V0KGEuZGF0YUlkLCB7XG4gICAgICAgIGJhY2tlbmQ6IGJhY2tlbmQgfHwgdGhpcy5iYWNrZW5kLFxuICAgICAgICBkdHlwZTogYS5kdHlwZSxcbiAgICAgICAgc2hhcGU6IGEuc2hhcGUsXG4gICAgICAgIGJ5dGVzXG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBpZiAoIShhIGluc3RhbmNlb2YgVmFyaWFibGUpKSB7XG4gICAgICB0aGlzLnRyYWNrKGEpO1xuICAgIH1cbiAgfVxuXG4gIC8vIFRyYWNrIHRoZSB0ZW5zb3IgYnkgZGF0YUlkIGFuZCBpbmNyZWFzZSB0aGUgcmVmQ291bnQgZm9yIHRoZSBkYXRhSWQgaW4gdGhlXG4gIC8vIGJhY2tlbmQuXG4gIC8vIFRPRE8ocHl1MTAwNTUpOiBUaGlzIGlzIGN1cnJlbnRseSB1c2VkIGJ5IG1ha2VWYXJpYWJsZSBtZXRob2QsIHRvIGluY3JlYXNlXG4gIC8vIHJlZkNvdW50IG9uIHRoZSBiYWNrZW5kIGZvciB0aGUgZGF0YUlkLiBJdCBjYW4gcG90ZW50aWFsbHkgYmUgcmVwbGFjZWQgd2l0aFxuICAvLyBJZGVudGl0eSBvcCBpbmRlYWQgb2YgY2FsbGluZyBiYWNrZW5kIGRpcmVjdGx5LlxuICBpbmNSZWYoYTogVGVuc29yLCBiYWNrZW5kOiBLZXJuZWxCYWNrZW5kKTogdm9pZCB7XG4gICAgdGhpcy50cmFja1RlbnNvcihhLCBiYWNrZW5kKTtcbiAgICB0aGlzLmJhY2tlbmQuaW5jUmVmKGEuZGF0YUlkKTtcbiAgfVxuXG4gIHJlbW92ZURhdGFJZChkYXRhSWQ6IERhdGFJZCwgYmFja2VuZDogS2VybmVsQmFja2VuZCkge1xuICAgIGlmICh0aGlzLnN0YXRlLnRlbnNvckluZm8uaGFzKGRhdGFJZCkgJiZcbiAgICAgICAgdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChkYXRhSWQpLmJhY2tlbmQgPT09IGJhY2tlbmQpIHtcbiAgICAgIHRoaXMuc3RhdGUudGVuc29ySW5mby5kZWxldGUoZGF0YUlkKTtcbiAgICAgIHRoaXMuc3RhdGUubnVtRGF0YUJ1ZmZlcnMtLTtcbiAgICB9XG4gIH1cbiAgZGlzcG9zZVRlbnNvcihhOiBUZW5zb3IpOiB2b2lkIHtcbiAgICBpZiAoIXRoaXMuc3RhdGUudGVuc29ySW5mby5oYXMoYS5kYXRhSWQpKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGEuZGF0YUlkKTtcblxuICAgIHRoaXMuc3RhdGUubnVtVGVuc29ycy0tO1xuICAgIGlmIChhLmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgdGhpcy5zdGF0ZS5udW1TdHJpbmdUZW5zb3JzLS07XG4gICAgICB0aGlzLnN0YXRlLm51bUJ5dGVzIC09IGluZm8uYnl0ZXM7XG4gICAgfVxuICAgIC8vIERvbid0IGNvdW50IGJ5dGVzIGZvciBjb21wbGV4IG51bWJlcnMgYXMgdGhleSBhcmUgY291bnRlZCBieSB0aGVpclxuICAgIC8vIGNvbXBvbmVudHMuXG4gICAgaWYgKGEuZHR5cGUgIT09ICdjb21wbGV4NjQnICYmIGEuZHR5cGUgIT09ICdzdHJpbmcnKSB7XG4gICAgICBjb25zdCBieXRlcyA9IGEuc2l6ZSAqIHV0aWwuYnl0ZXNQZXJFbGVtZW50KGEuZHR5cGUpO1xuICAgICAgdGhpcy5zdGF0ZS5udW1CeXRlcyAtPSBieXRlcztcbiAgICB9XG5cbiAgICAvLyBSZW1vdmUgdGhlIHJlZmVyZW5jZSB0byBkYXRhSWQgaWYgYmFja2VuZCBkaXNwb3NlIHRoZSBkYXRhIHN1Y2Nlc3NmdWxseVxuICAgIGlmIChpbmZvLmJhY2tlbmQuZGlzcG9zZURhdGEoYS5kYXRhSWQpKSB7XG4gICAgICB0aGlzLnJlbW92ZURhdGFJZChhLmRhdGFJZCwgaW5mby5iYWNrZW5kKTtcbiAgICB9XG5cbiAgICAvLyBUT0RPKG5zdGhvcmF0KTogQ29uc3RydWN0IGFuIGVycm9yIGFuZCBzYXZlIHRoZSBzdGFjayB0cmFjZSBmb3JcbiAgICAvLyBkZWJ1Z2dpbmcgd2hlbiBpbiBkZWJ1ZyBtb2RlLiBDcmVhdGluZyBhIHN0YWNrIHRyYWNlIGlzIHRvbyBleHBlbnNpdmVcbiAgICAvLyB0byBkbyB1bmNvbmRpdGlvbmFsbHkuXG4gIH1cblxuICBkaXNwb3NlVmFyaWFibGVzKCk6IHZvaWQge1xuICAgIGZvciAoY29uc3QgdmFyTmFtZSBpbiB0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXMpIHtcbiAgICAgIGNvbnN0IHYgPSB0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdmFyTmFtZV07XG4gICAgICB0aGlzLmRpc3Bvc2VWYXJpYWJsZSh2KTtcbiAgICB9XG4gIH1cblxuICBkaXNwb3NlVmFyaWFibGUodjogVmFyaWFibGUpOiB2b2lkIHtcbiAgICB0aGlzLmRpc3Bvc2VUZW5zb3Iodik7XG4gICAgaWYgKHRoaXMuc3RhdGUucmVnaXN0ZXJlZFZhcmlhYmxlc1t2Lm5hbWVdICE9IG51bGwpIHtcbiAgICAgIGRlbGV0ZSB0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXNbdi5uYW1lXTtcbiAgICB9XG4gIH1cblxuICBtZW1vcnkoKTogTWVtb3J5SW5mbyB7XG4gICAgY29uc3QgaW5mbyA9IHRoaXMuYmFja2VuZC5tZW1vcnkoKSBhcyBNZW1vcnlJbmZvO1xuICAgIGluZm8ubnVtVGVuc29ycyA9IHRoaXMuc3RhdGUubnVtVGVuc29ycztcbiAgICBpbmZvLm51bURhdGFCdWZmZXJzID0gdGhpcy5zdGF0ZS5udW1EYXRhQnVmZmVycztcbiAgICBpbmZvLm51bUJ5dGVzID0gdGhpcy5zdGF0ZS5udW1CeXRlcztcbiAgICBpZiAodGhpcy5zdGF0ZS5udW1TdHJpbmdUZW5zb3JzID4gMCkge1xuICAgICAgaW5mby51bnJlbGlhYmxlID0gdHJ1ZTtcbiAgICAgIGlmIChpbmZvLnJlYXNvbnMgPT0gbnVsbCkge1xuICAgICAgICBpbmZvLnJlYXNvbnMgPSBbXTtcbiAgICAgIH1cbiAgICAgIGluZm8ucmVhc29ucy5wdXNoKFxuICAgICAgICAgICdNZW1vcnkgdXNhZ2UgYnkgc3RyaW5nIHRlbnNvcnMgaXMgYXBwcm94aW1hdGUgJyArXG4gICAgICAgICAgJygyIGJ5dGVzIHBlciBjaGFyYWN0ZXIpJyk7XG4gICAgfVxuICAgIHJldHVybiBpbmZvO1xuICB9XG5cbiAgYXN5bmMgcHJvZmlsZShxdWVyeTogKCkgPT4gKFRlbnNvckNvbnRhaW5lciB8IFByb21pc2U8VGVuc29yQ29udGFpbmVyPikpOlxuICAgICAgUHJvbWlzZTxQcm9maWxlSW5mbz4ge1xuICAgIHRoaXMuc3RhdGUucHJvZmlsaW5nID0gdHJ1ZTtcblxuICAgIGNvbnN0IHN0YXJ0Qnl0ZXMgPSB0aGlzLnN0YXRlLm51bUJ5dGVzO1xuICAgIGNvbnN0IHN0YXJ0TnVtVGVuc29ycyA9IHRoaXMuc3RhdGUubnVtVGVuc29ycztcblxuICAgIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5rZXJuZWxzID0gW107XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLnJlc3VsdCA9IGF3YWl0IHF1ZXJ5KCk7XG5cbiAgICB0aGlzLnN0YXRlLnByb2ZpbGluZyA9IGZhbHNlO1xuXG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlLnBlYWtCeXRlcyA9IE1hdGgubWF4KFxuICAgICAgICAuLi50aGlzLnN0YXRlLmFjdGl2ZVByb2ZpbGUua2VybmVscy5tYXAoZCA9PiBkLnRvdGFsQnl0ZXNTbmFwc2hvdCkpO1xuICAgIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5uZXdCeXRlcyA9IHRoaXMuc3RhdGUubnVtQnl0ZXMgLSBzdGFydEJ5dGVzO1xuICAgIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5uZXdUZW5zb3JzID1cbiAgICAgICAgdGhpcy5zdGF0ZS5udW1UZW5zb3JzIC0gc3RhcnROdW1UZW5zb3JzO1xuICAgIGZvciAoY29uc3Qga2VybmVsIG9mIHRoaXMuc3RhdGUuYWN0aXZlUHJvZmlsZS5rZXJuZWxzKSB7XG4gICAgICBrZXJuZWwua2VybmVsVGltZU1zID0gYXdhaXQga2VybmVsLmtlcm5lbFRpbWVNcztcbiAgICAgIGtlcm5lbC5leHRyYUluZm8gPSBhd2FpdCBrZXJuZWwuZXh0cmFJbmZvO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5zdGF0ZS5hY3RpdmVQcm9maWxlO1xuICB9XG5cbiAgaXNUYXBlT24oKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHRoaXMuc3RhdGUuZ3JhZGllbnREZXB0aCA+IDAgJiYgdGhpcy5zdGF0ZS5rZXJuZWxEZXB0aCA9PT0gMDtcbiAgfVxuXG4gIHByaXZhdGUgYWRkVGFwZU5vZGUoXG4gICAgICBrZXJuZWxOYW1lOiBzdHJpbmcsIGlucHV0czogTmFtZWRUZW5zb3JNYXAsIG91dHB1dHM6IFRlbnNvcltdLFxuICAgICAgZ3JhZGllbnRzRnVuYzogR3JhZEZ1bmMsIHNhdmVkOiBUZW5zb3JbXSwgYXR0cnM6IE5hbWVkQXR0ck1hcCk6IHZvaWQge1xuICAgIGNvbnN0IHRhcGVOb2RlOiBUYXBlTm9kZSA9XG4gICAgICAgIHtpZDogdGhpcy5zdGF0ZS5uZXh0VGFwZU5vZGVJZCsrLCBrZXJuZWxOYW1lLCBpbnB1dHMsIG91dHB1dHMsIHNhdmVkfTtcblxuICAgIGNvbnN0IGdyYWRDb25maWcgPSBnZXRHcmFkaWVudChrZXJuZWxOYW1lKTtcbiAgICBpZiAoZ3JhZENvbmZpZyAhPSBudWxsKSB7XG4gICAgICBncmFkaWVudHNGdW5jID0gZ3JhZENvbmZpZy5ncmFkRnVuYztcbiAgICB9XG4gICAgaWYgKGdyYWRpZW50c0Z1bmMgIT0gbnVsbCkge1xuICAgICAgdGFwZU5vZGUuZ3JhZGllbnQgPSAoZHlzOiBUZW5zb3JbXSkgPT4ge1xuICAgICAgICAvLyBUT0RPKHNtaWxrb3YpOiBUbyBvcHRpbWl6ZSBiYWNrLXByb3AsIHBhc3MgZHlzIHRoYXQgYXJlIG5vdCB1c2VkIGluXG4gICAgICAgIC8vIHRoZSBiYWNrcHJvcCBncmFwaCB0byB0aGUgdXNlciBhcyBudWxsIGluc3RlYWQgb2YgemVyb3NcbiAgICAgICAgZHlzID0gZHlzLm1hcCgoZHksIGkpID0+IHtcbiAgICAgICAgICBpZiAoZHkgPT0gbnVsbCkge1xuICAgICAgICAgICAgY29uc3Qgb3V0cHV0ID0gb3V0cHV0c1tpXTtcbiAgICAgICAgICAgIGNvbnN0IHZhbHMgPSB1dGlsLm1ha2VaZXJvc1R5cGVkQXJyYXkob3V0cHV0LnNpemUsIG91dHB1dC5kdHlwZSk7XG4gICAgICAgICAgICByZXR1cm4gdGhpcy5tYWtlVGVuc29yKHZhbHMsIG91dHB1dC5zaGFwZSwgb3V0cHV0LmR0eXBlKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgcmV0dXJuIGR5O1xuICAgICAgICB9KTtcbiAgICAgICAgLy8gR3JhZCBmdW5jdGlvbnMgb2Ygb3BzIHdpdGggc2luZ2xlIG91dHB1dHMgZXhwZWN0IGEgZHksIHdoaWxlIG9wc1xuICAgICAgICAvLyB3aXRoIG11bHRpcGxlIG91dHB1dHMgZXhwZWN0IGR5cyAoYXJyYXkgb2YgZHkpLlxuICAgICAgICByZXR1cm4gZ3JhZGllbnRzRnVuYyhkeXMubGVuZ3RoID4gMSA/IGR5cyA6IGR5c1swXSwgc2F2ZWQsIGF0dHJzKTtcbiAgICAgIH07XG4gICAgfVxuICAgIHRoaXMuc3RhdGUuYWN0aXZlVGFwZS5wdXNoKHRhcGVOb2RlKTtcbiAgfVxuXG4gIGtlZXA8VCBleHRlbmRzIFRlbnNvcj4ocmVzdWx0OiBUKTogVCB7XG4gICAgcmVzdWx0LmtlcHQgPSB0cnVlO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIHN0YXJ0VGFwZSgpIHtcbiAgICBpZiAodGhpcy5zdGF0ZS5ncmFkaWVudERlcHRoID09PSAwKSB7XG4gICAgICB0aGlzLnN0YXRlLmFjdGl2ZVRhcGUgPSBbXTtcbiAgICB9XG4gICAgdGhpcy5zdGF0ZS5ncmFkaWVudERlcHRoKys7XG4gIH1cblxuICBwcml2YXRlIGVuZFRhcGUoKSB7XG4gICAgdGhpcy5zdGF0ZS5ncmFkaWVudERlcHRoLS07XG4gIH1cblxuICAvKipcbiAgICogU3RhcnQgYSBzY29wZS4gVXNlIHRoaXMgd2l0aCBlbmRTY29wZSgpIHRvIGFjaGlldmUgdGhlIHNhbWUgZnVuY3Rpb25hbGl0eVxuICAgKiBhcyBzY29wZSgpIHdpdGhvdXQgdGhlIG5lZWQgZm9yIGEgZnVuY3Rpb24gY2xvc3VyZS5cbiAgICovXG4gIHN0YXJ0U2NvcGUobmFtZT86IHN0cmluZykge1xuICAgIGNvbnN0IHNjb3BlSW5mbzogU2NvcGVTdGF0ZSA9IHtcbiAgICAgIHRyYWNrOiBbXSxcbiAgICAgIG5hbWU6ICd1bm5hbWVkIHNjb3BlJyxcbiAgICAgIGlkOiB0aGlzLnN0YXRlLm5leHRTY29wZUlkKytcbiAgICB9O1xuICAgIGlmIChuYW1lKSB7XG4gICAgICBzY29wZUluZm8ubmFtZSA9IG5hbWU7XG4gICAgfVxuICAgIHRoaXMuc3RhdGUuc2NvcGVTdGFjay5wdXNoKHNjb3BlSW5mbyk7XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZSA9IHNjb3BlSW5mbztcbiAgfVxuXG4gIC8qKlxuICAgKiBFbmQgYSBzY29wZS4gVXNlIHRoaXMgd2l0aCBzdGFydFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgZW5kU2NvcGUocmVzdWx0PzogVGVuc29yQ29udGFpbmVyKSB7XG4gICAgY29uc3QgdGVuc29yc1RvVHJhY2tJblBhcmVudCA9IGdldFRlbnNvcnNJbkNvbnRhaW5lcihyZXN1bHQpO1xuICAgIGNvbnN0IHRlbnNvcnNUb1RyYWNrSW5QYXJlbnRTZXQgPVxuICAgICAgICBuZXcgU2V0KHRlbnNvcnNUb1RyYWNrSW5QYXJlbnQubWFwKHQgPT4gdC5pZCkpO1xuXG4gICAgLy8gRGlzcG9zZSB0aGUgYXJyYXlzIHRyYWNrZWQgaW4gdGhpcyBzY29wZS5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUudHJhY2subGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IHRlbnNvciA9IHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUudHJhY2tbaV07XG4gICAgICBpZiAoIXRlbnNvci5rZXB0ICYmICF0ZW5zb3JzVG9UcmFja0luUGFyZW50U2V0Lmhhcyh0ZW5zb3IuaWQpKSB7XG4gICAgICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3Qgb2xkU2NvcGUgPSB0aGlzLnN0YXRlLnNjb3BlU3RhY2sucG9wKCk7XG4gICAgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZSA9IHRoaXMuc3RhdGUuc2NvcGVTdGFjay5sZW5ndGggPT09IDAgP1xuICAgICAgICBudWxsIDpcbiAgICAgICAgdGhpcy5zdGF0ZS5zY29wZVN0YWNrW3RoaXMuc3RhdGUuc2NvcGVTdGFjay5sZW5ndGggLSAxXTtcblxuICAgIC8vIFRyYWNrIHRoZSBjdXJyZW50IHJlc3VsdCBpbiB0aGUgcGFyZW50IHNjb3BlLlxuICAgIHRlbnNvcnNUb1RyYWNrSW5QYXJlbnQuZm9yRWFjaCh0ZW5zb3IgPT4ge1xuICAgICAgLy8gT25seSB0cmFjayB0aGUgdGVuc29yIGlmIHdhcyBhbGxvY2F0ZWQgaW4gdGhlIGlubmVyIHNjb3BlIGFuZCBpcyBub3RcbiAgICAgIC8vIGdsb2JhbGx5IGtlcHQuXG4gICAgICBpZiAoIXRlbnNvci5rZXB0ICYmIHRlbnNvci5zY29wZUlkID09PSBvbGRTY29wZS5pZCkge1xuICAgICAgICB0aGlzLnRyYWNrKHRlbnNvcik7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJucyBncmFkaWVudHMgb2YgYGZgIHdpdGggcmVzcGVjdCB0byBlYWNoIG9mIHRoZSBgeHNgLiBUaGUgZ3JhZGllbnRzXG4gICAqIHJldHVybmVkIGFyZSBvZiB0aGUgc2FtZSBsZW5ndGggYXMgYHhzYCwgYnV0IHNvbWUgbWlnaHQgYmUgbnVsbCBpZiBgZmBcbiAgICogd2FzIG5vdCBhIGZ1bmN0aW9uIG9mIHRoYXQgYHhgLiBJdCBhbHNvIHRha2VzIG9wdGlvbmFsIGR5IHRvIG11bHRpcGx5IHRoZVxuICAgKiBncmFkaWVudCwgd2hpY2ggZGVmYXVsdHMgdG8gYDFgLlxuICAgKi9cbiAgZ3JhZGllbnRzPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgICAgZjogKCkgPT4gVCwgeHM6IFRlbnNvcltdLCBkeT86IFQsXG4gICAgICBhbGxvd05vR3JhZGllbnRzID0gZmFsc2UpOiB7dmFsdWU6IFQsIGdyYWRzOiBUZW5zb3JbXX0ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4cy5sZW5ndGggPiAwLCAoKSA9PiAnZ3JhZGllbnRzKCkgcmVjZWl2ZWQgYW4gZW1wdHkgbGlzdCBvZiB4cy4nKTtcbiAgICBpZiAoZHkgIT0gbnVsbCAmJiBkeS5kdHlwZSAhPT0gJ2Zsb2F0MzInKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYGR5IG11c3QgaGF2ZSAnZmxvYXQzMicgZHR5cGUsIGJ1dCBoYXMgJyR7ZHkuZHR5cGV9J2ApO1xuICAgIH1cblxuICAgIGNvbnN0IHkgPSB0aGlzLnNjb3BlZFJ1bihcbiAgICAgICAgKCkgPT4gdGhpcy5zdGFydFRhcGUoKSwgKCkgPT4gdGhpcy5lbmRUYXBlKCksXG4gICAgICAgICgpID0+IHRoaXMudGlkeSgnZm9yd2FyZCcsIGYpKTtcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB5IGluc3RhbmNlb2YgVGVuc29yLFxuICAgICAgICAoKSA9PiAnVGhlIHJlc3VsdCB5IHJldHVybmVkIGJ5IGYoKSBtdXN0IGJlIGEgdGVuc29yLicpO1xuICAgIC8vIEZpbHRlciBvdXQgdGhlIG5vZGVzIHRoYXQgZG9uJ3QgY29ubmVjdCB4ID0+IHkuXG4gICAgY29uc3QgZmlsdGVyZWRUYXBlID0gZ2V0RmlsdGVyZWROb2Rlc1hUb1kodGhpcy5zdGF0ZS5hY3RpdmVUYXBlLCB4cywgeSk7XG4gICAgaWYgKCFhbGxvd05vR3JhZGllbnRzICYmIGZpbHRlcmVkVGFwZS5sZW5ndGggPT09IDAgJiYgeHMubGVuZ3RoID4gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICdDYW5ub3QgY29tcHV0ZSBncmFkaWVudCBvZiB5PWYoeCkgd2l0aCByZXNwZWN0IHRvIHguIE1ha2Ugc3VyZSAnICtcbiAgICAgICAgICAndGhhdCB0aGUgZiB5b3UgcGFzc2VkIGVuY2xvc2VzIGFsbCBvcGVyYXRpb25zIHRoYXQgbGVhZCBmcm9tIHggJyArXG4gICAgICAgICAgJ3RvIHkuJyk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHRoaXMudGlkeSgnYmFja3dhcmQnLCAoKSA9PiB7XG4gICAgICBjb25zdCBhY2N1bXVsYXRlZEdyYWRpZW50TWFwOiB7W3RlbnNvcklkOiBudW1iZXJdOiBUZW5zb3J9ID0ge307XG4gICAgICBhY2N1bXVsYXRlZEdyYWRpZW50TWFwW3kuaWRdID0gKGR5ID09IG51bGwpID8gb25lcyh5LnNoYXBlKSA6IGR5O1xuXG4gICAgICAvLyBCYWNrcHJvcCBncmFkaWVudHMgdGhyb3VnaCB0aGUgZmlsdGVyZWQgbm9kZXMuXG4gICAgICBiYWNrcHJvcGFnYXRlR3JhZGllbnRzKFxuICAgICAgICAgIGFjY3VtdWxhdGVkR3JhZGllbnRNYXAsIGZpbHRlcmVkVGFwZSxcbiAgICAgICAgICAvLyBQYXNzIHRoZSB0aWR5IGZ1bmN0aW9uIHRvIGF2b2lkIGNpcmN1bGFyIGRlcCB3aXRoIGB0YXBlLnRzYC5cbiAgICAgICAgICBmID0+IHRoaXMudGlkeShmIGFzIFNjb3BlRm48VGVuc29yPiksXG4gICAgICAgICAgLy8gUGFzcyBhbiBhZGQgZnVuY3Rpb24gdG8gYXZvaWRlIGEgY2lyY3VsYXIgZGVwIHdpdGggYHRhcGUudHNgLlxuICAgICAgICAgIGFkZCk7XG4gICAgICBjb25zdCBncmFkcyA9IHhzLm1hcCh4ID0+IGFjY3VtdWxhdGVkR3JhZGllbnRNYXBbeC5pZF0pO1xuXG4gICAgICBpZiAodGhpcy5zdGF0ZS5ncmFkaWVudERlcHRoID09PSAwKSB7XG4gICAgICAgIC8vIFRoaXMgbWVhbnMgdGhhdCB3ZSBhcmUgbm90IGNvbXB1dGluZyBoaWdoZXItb3JkZXIgZ3JhZGllbnRzXG4gICAgICAgIC8vIGFuZCBjYW4gY2xlYW4gdXAgdGhlIHRhcGUuXG4gICAgICAgIHRoaXMuc3RhdGUuYWN0aXZlVGFwZS5mb3JFYWNoKG5vZGUgPT4ge1xuICAgICAgICAgIGZvciAoY29uc3QgdGVuc29yIG9mIG5vZGUuc2F2ZWQpIHtcbiAgICAgICAgICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgICAgICAgfVxuICAgICAgICB9KTtcbiAgICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVUYXBlID0gbnVsbDtcbiAgICAgIH1cbiAgICAgIHJldHVybiB7dmFsdWU6IHksIGdyYWRzfTtcbiAgICB9KTtcbiAgfVxuXG4gIGN1c3RvbUdyYWQ8VCBleHRlbmRzIFRlbnNvcj4oZjogQ3VzdG9tR3JhZGllbnRGdW5jPFQ+KTpcbiAgICAgICguLi5hcmdzOiBBcnJheTxUZW5zb3J8R3JhZFNhdmVGdW5jPikgPT4gVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHV0aWwuaXNGdW5jdGlvbihmKSxcbiAgICAgICAgKCkgPT4gJ1RoZSBmIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpIG11c3QgYmUgYSBmdW5jdGlvbi4nKTtcbiAgICByZXR1cm4gKC4uLmlucHV0czogVGVuc29yW10pOiBUID0+IHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGlucHV0cy5ldmVyeSh0ID0+IHQgaW5zdGFuY2VvZiBUZW5zb3IpLFxuICAgICAgICAgICgpID0+ICdUaGUgYXJncyBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSh4MSwgeDIsLi4uKSBtdXN0IGFsbCBiZSAnICtcbiAgICAgICAgICAgICAgJ3RlbnNvcnMnKTtcblxuICAgICAgbGV0IHJlczoge1xuICAgICAgICB2YWx1ZTogVCxcbiAgICAgICAgZ3JhZEZ1bmM6IChkeTogVCwgc2F2ZWQ6IFRlbnNvcltdKSA9PiBUZW5zb3IgfCBUZW5zb3JbXSxcbiAgICAgIH07XG4gICAgICBjb25zdCBpbnB1dE1hcDogTmFtZWRUZW5zb3JNYXAgPSB7fTtcbiAgICAgIGlucHV0cy5mb3JFYWNoKChpbnB1dCwgaSkgPT4ge1xuICAgICAgICBpbnB1dE1hcFtpXSA9IGlucHV0O1xuICAgICAgfSk7XG5cbiAgICAgIGNvbnN0IGZvcndhcmRGdW5jOiBGb3J3YXJkRnVuYzxUPiA9IChfLCBzYXZlKSA9PiB7XG4gICAgICAgIHJlcyA9IGYoLi4uWy4uLmlucHV0cywgc2F2ZV0pO1xuICAgICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAgIHJlcy52YWx1ZSBpbnN0YW5jZW9mIFRlbnNvcixcbiAgICAgICAgICAgICgpID0+ICdUaGUgZnVuY3Rpb24gZiBwYXNzZWQgaW4gY3VzdG9tR3JhZChmKSBtdXN0IHJldHVybiBhbiAnICtcbiAgICAgICAgICAgICAgICAnb2JqZWN0IHdoZXJlIGBvYmoudmFsdWVgIGlzIGEgdGVuc29yJyk7XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgdXRpbC5pc0Z1bmN0aW9uKHJlcy5ncmFkRnVuYyksXG4gICAgICAgICAgICAoKSA9PiAnVGhlIGZ1bmN0aW9uIGYgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikgbXVzdCByZXR1cm4gYW4gJyArXG4gICAgICAgICAgICAgICAgJ29iamVjdCB3aGVyZSBgb2JqLmdyYWRGdW5jYCBpcyBhIGZ1bmN0aW9uLicpO1xuICAgICAgICByZXR1cm4gcmVzLnZhbHVlO1xuICAgICAgfTtcblxuICAgICAgY29uc3QgYmFja3dhcmRzRnVuYyA9IChkeTogVCwgc2F2ZWQ6IFRlbnNvcltdKSA9PiB7XG4gICAgICAgIGNvbnN0IGdyYWRSZXMgPSByZXMuZ3JhZEZ1bmMoZHksIHNhdmVkKTtcbiAgICAgICAgY29uc3QgZ3JhZHM6IFRlbnNvcltdID0gQXJyYXkuaXNBcnJheShncmFkUmVzKSA/IGdyYWRSZXMgOiBbZ3JhZFJlc107XG4gICAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgICAgZ3JhZHMubGVuZ3RoID09PSBpbnB1dHMubGVuZ3RoLFxuICAgICAgICAgICAgKCkgPT4gJ1RoZSBmdW5jdGlvbiBmIHBhc3NlZCBpbiBjdXN0b21HcmFkKGYpIG11c3QgcmV0dXJuIGFuICcgK1xuICAgICAgICAgICAgICAgICdvYmplY3Qgd2hlcmUgYG9iai5ncmFkRnVuY2AgaXMgYSBmdW5jdGlvbiB0aGF0IHJldHVybnMgJyArXG4gICAgICAgICAgICAgICAgJ3RoZSBzYW1lIG51bWJlciBvZiB0ZW5zb3JzIGFzIGlucHV0cyBwYXNzZWQgdG8gZiguLi4pLicpO1xuICAgICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAgIGdyYWRzLmV2ZXJ5KHQgPT4gdCBpbnN0YW5jZW9mIFRlbnNvciksXG4gICAgICAgICAgICAoKSA9PiAnVGhlIGZ1bmN0aW9uIGYgcGFzc2VkIGluIGN1c3RvbUdyYWQoZikgbXVzdCByZXR1cm4gYW4gJyArXG4gICAgICAgICAgICAgICAgJ29iamVjdCB3aGVyZSBgb2JqLmdyYWRGdW5jYCBpcyBhIGZ1bmN0aW9uIHRoYXQgcmV0dXJucyAnICtcbiAgICAgICAgICAgICAgICAnYSBsaXN0IG9mIG9ubHkgdGVuc29ycy4nKTtcbiAgICAgICAgY29uc3QgZ3JhZE1hcDoge1trZXk6IHN0cmluZ106ICgpID0+IFRlbnNvcn0gPSB7fTtcbiAgICAgICAgZ3JhZHMuZm9yRWFjaCgoZ3JhZCwgaSkgPT4ge1xuICAgICAgICAgIGdyYWRNYXBbaV0gPSAoKSA9PiBncmFkO1xuICAgICAgICB9KTtcbiAgICAgICAgcmV0dXJuIGdyYWRNYXA7XG4gICAgICB9O1xuXG4gICAgICByZXR1cm4gdGhpcy5ydW5LZXJuZWxGdW5jKHtcbiAgICAgICAgZm9yd2FyZEZ1bmMsXG4gICAgICAgIGJhY2t3YXJkc0Z1bmMsXG4gICAgICAgIGlucHV0czogaW5wdXRNYXAsXG4gICAgICB9KTtcbiAgICB9O1xuICB9XG5cbiAgcmVhZFN5bmMoZGF0YUlkOiBEYXRhSWQpOiBCYWNrZW5kVmFsdWVzIHtcbiAgICAvLyBSb3V0ZSB0aGUgcmVhZCB0byB0aGUgY29ycmVjdCBiYWNrZW5kLlxuICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCk7XG4gICAgcmV0dXJuIGluZm8uYmFja2VuZC5yZWFkU3luYyhkYXRhSWQpO1xuICB9XG4gIHJlYWQoZGF0YUlkOiBEYXRhSWQpOiBQcm9taXNlPEJhY2tlbmRWYWx1ZXM+IHtcbiAgICAvLyBSb3V0ZSB0aGUgcmVhZCB0byB0aGUgY29ycmVjdCBiYWNrZW5kLlxuICAgIGNvbnN0IGluZm8gPSB0aGlzLnN0YXRlLnRlbnNvckluZm8uZ2V0KGRhdGFJZCk7XG4gICAgcmV0dXJuIGluZm8uYmFja2VuZC5yZWFkKGRhdGFJZCk7XG4gIH1cblxuICByZWFkVG9HUFUoZGF0YUlkOiBEYXRhSWQsIG9wdGlvbnM/OiBEYXRhVG9HUFVPcHRpb25zKTogR1BVRGF0YSB7XG4gICAgLy8gUm91dGUgdGhlIHJlYWQgdG8gdGhlIGNvcnJlY3QgYmFja2VuZC5cbiAgICBjb25zdCBpbmZvID0gdGhpcy5zdGF0ZS50ZW5zb3JJbmZvLmdldChkYXRhSWQpO1xuICAgIHJldHVybiBpbmZvLmJhY2tlbmQucmVhZFRvR1BVKGRhdGFJZCwgb3B0aW9ucyk7XG4gIH1cblxuICBhc3luYyB0aW1lKHF1ZXJ5OiAoKSA9PiB2b2lkKTogUHJvbWlzZTxUaW1pbmdJbmZvPiB7XG4gICAgY29uc3Qgc3RhcnQgPSBub3coKTtcbiAgICBjb25zdCB0aW1pbmdJbmZvID0gYXdhaXQgdGhpcy5iYWNrZW5kLnRpbWUocXVlcnkpIGFzIFRpbWluZ0luZm87XG4gICAgdGltaW5nSW5mby53YWxsTXMgPSBub3coKSAtIHN0YXJ0O1xuICAgIHJldHVybiB0aW1pbmdJbmZvO1xuICB9XG5cbiAgLyoqXG4gICAqIFRyYWNrcyBhIFRlbnNvciBpbiB0aGUgY3VycmVudCBzY29wZSB0byBiZSBhdXRvbWF0aWNhbGx5IGNsZWFuZWQgdXBcbiAgICogd2hlbiB0aGUgY3VycmVudCBzY29wZSBlbmRzLCBhbmQgcmV0dXJucyB0aGUgdmFsdWUuXG4gICAqXG4gICAqIEBwYXJhbSByZXN1bHQgVGhlIFRlbnNvciB0byB0cmFjayBpbiB0aGUgY3VycmVudCBzY29wZS5cbiAgICovXG4gIHByaXZhdGUgdHJhY2s8VCBleHRlbmRzIFRlbnNvcj4ocmVzdWx0OiBUKTogVCB7XG4gICAgaWYgKHRoaXMuc3RhdGUuYWN0aXZlU2NvcGUgIT0gbnVsbCkge1xuICAgICAgcmVzdWx0LnNjb3BlSWQgPSB0aGlzLnN0YXRlLmFjdGl2ZVNjb3BlLmlkO1xuICAgICAgdGhpcy5zdGF0ZS5hY3RpdmVTY29wZS50cmFjay5wdXNoKHJlc3VsdCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIGdldCByZWdpc3RlcmVkVmFyaWFibGVzKCk6IE5hbWVkVmFyaWFibGVNYXAge1xuICAgIHJldHVybiB0aGlzLnN0YXRlLnJlZ2lzdGVyZWRWYXJpYWJsZXM7XG4gIH1cblxuICAvKipcbiAgICogUmVzZXRzIHRoZSBlbmdpbmUgc3RhdGUuIFJlbW92ZXMgYWxsIGJhY2tlbmRzIGJ1dCBkb2VzIG5vdCByZW1vdmVcbiAgICogcmVnaXN0ZXJlZCBiYWNrZW5kIGZhY3Rvcmllcy5cbiAgICovXG4gIHJlc2V0KCk6IHZvaWQge1xuICAgIC8vIE1ha2UgYW55IHBlbmRpbmcgcHJvbWlzZSBvYnNvbGV0ZS5cbiAgICB0aGlzLnBlbmRpbmdCYWNrZW5kSW5pdElkKys7XG5cbiAgICB0aGlzLnN0YXRlLmRpc3Bvc2UoKTtcbiAgICB0aGlzLkVOVi5yZXNldCgpO1xuICAgIHRoaXMuc3RhdGUgPSBuZXcgRW5naW5lU3RhdGUoKTtcblxuICAgIGZvciAoY29uc3QgYmFja2VuZE5hbWUgaW4gdGhpcy5yZWdpc3RyeSkge1xuICAgICAgdGhpcy5kaXNwb3NlUmVnaXN0ZXJlZEtlcm5lbHMoYmFja2VuZE5hbWUpO1xuICAgICAgdGhpcy5yZWdpc3RyeVtiYWNrZW5kTmFtZV0uZGlzcG9zZSgpO1xuICAgICAgZGVsZXRlIHRoaXMucmVnaXN0cnlbYmFja2VuZE5hbWVdO1xuICAgIH1cbiAgICB0aGlzLmJhY2tlbmROYW1lID0gbnVsbDtcbiAgICB0aGlzLmJhY2tlbmRJbnN0YW5jZSA9IG51bGw7XG4gICAgdGhpcy5wZW5kaW5nQmFja2VuZEluaXQgPSBudWxsO1xuICB9XG59XG5cbmZ1bmN0aW9uIG9uZXMoc2hhcGU6IG51bWJlcltdKTogVGVuc29yIHtcbiAgY29uc3QgdmFsdWVzID0gbWFrZU9uZXNUeXBlZEFycmF5KHNpemVGcm9tU2hhcGUoc2hhcGUpLCAnZmxvYXQzMicpO1xuICByZXR1cm4gRU5HSU5FLm1ha2VUZW5zb3IodmFsdWVzLCBzaGFwZSwgJ2Zsb2F0MzInKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE9yTWFrZUVuZ2luZSgpOiBFbmdpbmUge1xuICBjb25zdCBucyA9IGdldEdsb2JhbE5hbWVzcGFjZSgpIGFzIHVua25vd24gYXMge190ZmVuZ2luZTogRW5naW5lfTtcbiAgaWYgKG5zLl90ZmVuZ2luZSA9PSBudWxsKSB7XG4gICAgY29uc3QgZW52aXJvbm1lbnQgPSBuZXcgRW52aXJvbm1lbnQobnMpO1xuICAgIG5zLl90ZmVuZ2luZSA9IG5ldyBFbmdpbmUoZW52aXJvbm1lbnQpO1xuICB9XG4gIHNldEVudmlyb25tZW50R2xvYmFsKG5zLl90ZmVuZ2luZS5FTlYpO1xuXG4gIC8vIFRlbGwgdGhlIGN1cnJlbnQgdGVuc29yIGludGVyZmFjZSB0aGF0IHRoZSBnbG9iYWwgZW5naW5lIGlzIHJlc3BvbnNpYmxlXG4gIC8vIGZvciB0cmFja2luZy5cbiAgc2V0VGVuc29yVHJhY2tlcigoKSA9PiBucy5fdGZlbmdpbmUpO1xuICByZXR1cm4gbnMuX3RmZW5naW5lO1xufVxuXG5leHBvcnQgY29uc3QgRU5HSU5FID0gZ2V0T3JNYWtlRW5naW5lKCk7XG5cbi8qKlxuICogQSBpbXBsZW1lbnRhdGlvbiBvZiB0aGUgYWRkIG9wIGZvciB1c2Ugd2l0aGluIGVuZ2luZSBhbmQgdGFwZS5cbiAqXG4gKiBUaGlzIGFsbG93cyB1cyB0byBhdm9pZCBhIGNpcmN1bGFyIGRlcGVuZGVuY3kgYmV0d2VlbiBhZGQudHMgYW5kIGVuZ2luZS5cbiAqIEl0IGlzIGV4cG9ydGVkIHRvIGJlIGF2YWlsYWJsZSBpbiB0YXBlIHRlc3RzLlxuICovXG5leHBvcnQgZnVuY3Rpb24gYWRkKGE6IFRlbnNvciwgYjogVGVuc29yKTogVGVuc29yIHtcbiAgLy8gV2UgZHVwbGljYXRlIEFkZCBoZXJlIHRvIGF2b2lkIGEgY2lyY3VsYXIgZGVwZW5kZW5jeSB3aXRoIGFkZC50cy5cbiAgY29uc3QgaW5wdXRzID0ge2EsIGJ9O1xuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChBZGQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwKTtcbn1cbiJdfQ==