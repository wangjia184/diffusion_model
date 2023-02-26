/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* Original source: keras/callbacks.py */
import { add, div, keep, mul, nextFrame, tidy, util } from '@tensorflow/tfjs-core';
import { ValueError } from './errors';
import { resolveScalarsInLogs } from './logs';
import * as generic_utils from './utils/generic_utils';
/** Verbosity logging level when fitting a model. */
export var ModelLoggingVerbosity;
(function (ModelLoggingVerbosity) {
    ModelLoggingVerbosity[ModelLoggingVerbosity["SILENT"] = 0] = "SILENT";
    ModelLoggingVerbosity[ModelLoggingVerbosity["VERBOSE"] = 1] = "VERBOSE";
})(ModelLoggingVerbosity || (ModelLoggingVerbosity = {}));
/** How often to yield to the main thread when training (in ms). */
export const DEFAULT_YIELD_EVERY_MS = 125;
/**
 * Abstract base class used to build new callbacks.
 *
 * The `logs` dictionary that callback methods take as argument will contain
 * keys for quantities relevant to the current batch or epoch.
 *
 * Currently, the `.fit()` method of the `Sequential` model class
 * will include the following quantities in the `logs` that
 * it passes to its callbacks:
 *
 * onEpochEnd: Logs include `acc` and `loss`, and optionally include `valLoss`
 *   (if validation is enabled in `fit`), and `valAcc` (if validation and
 *   accuracy monitoring are enabled).
 * onBatchBegin: Logs include `size`, the number of samples in the current
 *   batch.
 * onBatchEnd: Logs include `loss`, and optionally `acc` (if accuracy monitoring
 *   is enabled).
 */
export class BaseCallback {
    constructor() {
        // TODO(michaelterry): This type is a best guess.
        this.validationData = null;
    }
    setParams(params) {
        this.params = params;
    }
    async onEpochBegin(epoch, logs) { }
    async onEpochEnd(epoch, logs) { }
    async onBatchBegin(batch, logs) { }
    async onBatchEnd(batch, logs) { }
    async onTrainBegin(logs) { }
    async onTrainEnd(logs) { }
    // LayersModel needs to call Callback.setModel(), but cannot actually depend
    // on Callback because that creates a cyclic dependency.  Providing this no-op
    // method on BaseCallback breaks the cycle: this way LayersModel can depend on
    // BaseCallback but not on Callback.  The argument is typed as `Container`
    // (the superclass of LayersModel) to avoid recapitulating the cycle. Callback
    // overrides this method and enforces that the argument is really a
    // LayersModel.
    setModel(model) {
        // Do nothing. Use Callback instead of BaseCallback to track the model.
    }
}
/**
 * Container abstracting a list of callbacks.
 */
export class CallbackList {
    // TODO(cais): When the need arises, uncomment the following lines and
    // implement the queue for time values.
    // private deltaTBatch: number;
    // private deltaTsBatchBegin: Array<number>;
    // private deltaTsBatchEnd: Array<number>;
    /**
     * Constructor of CallbackList.
     * @param callbacks Array of `Callback` instances.
     * @param queueLength Queue length for keeping running statistics over
     *   callback execution time.
     */
    constructor(callbacks, queueLength = 10) {
        // TODO(cais): Make use of queueLength when implementing the queue for time
        // values.
        if (callbacks == null) {
            callbacks = [];
        }
        this.callbacks = callbacks;
        this.queueLength = queueLength;
    }
    append(callback) {
        this.callbacks.push(callback);
    }
    setParams(params) {
        for (const callback of this.callbacks) {
            callback.setParams(params);
        }
    }
    setModel(model) {
        for (const callback of this.callbacks) {
            callback.setModel(model);
        }
    }
    /**
     * Called at the start of an epoch.
     * @param epoch Index of epoch.
     * @param logs Dictionary of logs.
     */
    async onEpochBegin(epoch, logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onEpochBegin(epoch, logs);
        }
    }
    /**
     * Called at the end of an epoch.
     * @param epoch Index of epoch.
     * @param logs Dictionary of logs.
     */
    async onEpochEnd(epoch, logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onEpochEnd(epoch, logs);
        }
    }
    /**
     * Called  right before processing a batch.
     * @param batch Index of batch within the current epoch.
     * @param logs Dictionary of logs.
     */
    async onBatchBegin(batch, logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onBatchBegin(batch, logs);
        }
    }
    /**
     * Called at the end of a batch.
     * @param batch Index of batch within the current epoch.
     * @param logs Dictionary of logs.
     */
    async onBatchEnd(batch, logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onBatchEnd(batch, logs);
        }
    }
    /**
     * Called at the beginning of training.
     * @param logs Dictionary of logs.
     */
    async onTrainBegin(logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onTrainBegin(logs);
        }
    }
    /**
     * Called at the end of training.
     * @param logs Dictionary of logs.
     */
    async onTrainEnd(logs) {
        if (logs == null) {
            logs = {};
        }
        for (const callback of this.callbacks) {
            await callback.onTrainEnd(logs);
        }
    }
}
/**
 * Callback that accumulates epoch averages of metrics.
 *
 * This callback is automatically applied to every LayersModel.
 */
export class BaseLogger extends BaseCallback {
    constructor() {
        super();
    }
    async onEpochBegin(epoch) {
        this.seen = 0;
        this.totals = {};
    }
    async onBatchEnd(batch, logs) {
        if (logs == null) {
            logs = {};
        }
        const batchSize = logs['size'] == null ? 0 : logs['size'];
        this.seen += batchSize;
        for (const key in logs) {
            const value = logs[key];
            if (typeof value === 'number') {
                if (!this.totals.hasOwnProperty(key)) {
                    this.totals[key] = 0;
                }
                this.totals[key] = this.totals[key] + value * batchSize;
            }
            else {
                let oldTotalsToDispose;
                if (key in this.totals) {
                    oldTotalsToDispose = this.totals[key];
                }
                else {
                    this.totals[key] = 0;
                }
                const total = tidy(() => add((this.totals[key]), mul(value, batchSize)));
                this.totals[key] = total;
                if (oldTotalsToDispose != null) {
                    oldTotalsToDispose.dispose();
                }
            }
        }
    }
    async onEpochEnd(epoch, logs) {
        if (logs != null) {
            for (const key of this.params['metrics']) {
                if (this.totals[key] == null) {
                    continue;
                }
                if (typeof this.totals[key] === 'number') {
                    logs[key] = this.totals[key] / this.seen;
                }
                else {
                    tidy(() => {
                        const log = mul(div(1, this.seen), this.totals[key]);
                        logs[key] = log;
                        this.totals[key].dispose();
                        keep(logs[key]);
                    });
                }
            }
        }
    }
}
/**
 * Callback that records events into a `History` object. This callback is
 * automatically applied to every TF.js Layers model. The `History` object
 * gets returned by the `fit` method of models.
 */
export class History extends BaseCallback {
    async onTrainBegin(logs) {
        this.epoch = [];
        this.history = {};
    }
    async onEpochEnd(epoch, logs) {
        if (logs == null) {
            logs = {};
        }
        this.epoch.push(epoch);
        for (const key in logs) {
            if (this.history[key] == null) {
                this.history[key] = [];
            }
            this.history[key].push(logs[key]);
        }
    }
    /**
     * Await the values of all losses and metrics.
     */
    async syncData() {
        const promises = [];
        const keys = [];
        const indices = [];
        for (const key in this.history) {
            const valueArray = this.history[key];
            for (let i = 0; i < valueArray.length; ++i) {
                if (typeof valueArray[i] !== 'number') {
                    const valueScalar = valueArray[i];
                    promises.push(valueScalar.data());
                    keys.push(key);
                    indices.push(i);
                }
            }
        }
        const values = await Promise.all(promises);
        for (let n = 0; n < values.length; ++n) {
            const tensorToDispose = this.history[keys[n]][indices[n]];
            tensorToDispose.dispose();
            this.history[keys[n]][indices[n]] = values[n][0];
        }
    }
}
/**
 * Custom callback for training.
 */
export class CustomCallback extends BaseCallback {
    constructor(args, yieldEvery) {
        super();
        this.currentEpoch = 0;
        this.nowFunc = args.nowFunc;
        this.nextFrameFunc = args.nextFrameFunc || nextFrame;
        this.yieldEvery = yieldEvery || 'auto';
        if (this.yieldEvery === 'auto') {
            this.yieldEvery = DEFAULT_YIELD_EVERY_MS;
        }
        if (this.yieldEvery === 'never' && args.onYield != null) {
            throw new Error('yieldEvery is `never` but you provided an `onYield` callback. ' +
                'Either change `yieldEvery` or remove the callback');
        }
        if (util.isNumber(this.yieldEvery)) {
            // Decorate `maybeWait` so it will be called at most once every
            // `yieldEvery` ms.
            this.maybeWait = generic_utils.debounce(this.maybeWait.bind(this), this.yieldEvery, this.nowFunc);
        }
        this.trainBegin = args.onTrainBegin;
        this.trainEnd = args.onTrainEnd;
        this.epochBegin = args.onEpochBegin;
        this.epochEnd = args.onEpochEnd;
        this.batchBegin = args.onBatchBegin;
        this.batchEnd = args.onBatchEnd;
        this.yield = args.onYield;
    }
    async maybeWait(epoch, batch, logs) {
        const ps = [];
        if (this.yield != null) {
            await resolveScalarsInLogs(logs);
            ps.push(this.yield(epoch, batch, logs));
        }
        ps.push(this.nextFrameFunc());
        await Promise.all(ps);
    }
    async onEpochBegin(epoch, logs) {
        this.currentEpoch = epoch;
        if (this.epochBegin != null) {
            await resolveScalarsInLogs(logs);
            await this.epochBegin(epoch, logs);
        }
    }
    async onEpochEnd(epoch, logs) {
        const ps = [];
        if (this.epochEnd != null) {
            await resolveScalarsInLogs(logs);
            ps.push(this.epochEnd(epoch, logs));
        }
        if (this.yieldEvery === 'epoch') {
            ps.push(this.nextFrameFunc());
        }
        await Promise.all(ps);
    }
    async onBatchBegin(batch, logs) {
        if (this.batchBegin != null) {
            await resolveScalarsInLogs(logs);
            await this.batchBegin(batch, logs);
        }
    }
    async onBatchEnd(batch, logs) {
        const ps = [];
        if (this.batchEnd != null) {
            await resolveScalarsInLogs(logs);
            ps.push(this.batchEnd(batch, logs));
        }
        if (this.yieldEvery === 'batch') {
            ps.push(this.nextFrameFunc());
        }
        else if (util.isNumber(this.yieldEvery)) {
            ps.push(this.maybeWait(this.currentEpoch, batch, logs));
        }
        await Promise.all(ps);
    }
    async onTrainBegin(logs) {
        if (this.trainBegin != null) {
            await resolveScalarsInLogs(logs);
            await this.trainBegin(logs);
        }
    }
    async onTrainEnd(logs) {
        if (this.trainEnd != null) {
            await resolveScalarsInLogs(logs);
            await this.trainEnd(logs);
        }
    }
}
/**
 * Standardize callbacks or configurations of them to an Array of callbacks.
 */
export function standardizeCallbacks(callbacks, yieldEvery) {
    if (callbacks == null) {
        callbacks = {};
    }
    if (callbacks instanceof BaseCallback) {
        return [callbacks];
    }
    if (Array.isArray(callbacks) && callbacks[0] instanceof BaseCallback) {
        return callbacks;
    }
    // Convert custom callback configs to custom callback objects.
    const callbackConfigs = generic_utils.toList(callbacks);
    return callbackConfigs.map(callbackConfig => new CustomCallback(callbackConfig, yieldEvery));
}
/**
 * A global registry for callback constructors to be used during
 * LayersModel.fit().
 */
export class CallbackConstructorRegistry {
    /**
     * Blocks public access to constructor.
     */
    constructor() { }
    /**
     * Register a tf.LayersModel.fit() callback constructor.
     *
     * The registered callback constructor will be used to instantiate
     * callbacks for every tf.LayersModel.fit() call afterwards.
     *
     * @param verbosityLevel Level of verbosity at which the `callbackConstructor`
     *   is to be reigstered.
     * @param callbackConstructor A no-arg constructor for `tf.Callback`.
     * @throws Error, if the same callbackConstructor has been registered before,
     *   either at the same or a different `verbosityLevel`.
     */
    static registerCallbackConstructor(verbosityLevel, callbackConstructor) {
        util.assert(verbosityLevel >= 0 && Number.isInteger(verbosityLevel), () => `Verbosity level is expected to be an integer >= 0, ` +
            `but got ${verbosityLevel}`);
        CallbackConstructorRegistry.checkForDuplicate(callbackConstructor);
        if (CallbackConstructorRegistry.constructors[verbosityLevel] == null) {
            CallbackConstructorRegistry.constructors[verbosityLevel] = [];
        }
        CallbackConstructorRegistry.constructors[verbosityLevel].push(callbackConstructor);
    }
    static checkForDuplicate(callbackConstructor) {
        for (const levelName in CallbackConstructorRegistry.constructors) {
            const constructors = CallbackConstructorRegistry.constructors[+levelName];
            constructors.forEach(ctor => {
                if (ctor === callbackConstructor) {
                    throw new ValueError('Duplicate callback constructor.');
                }
            });
        }
    }
    /**
     * Clear all registered callback constructors.
     */
    static clear() {
        CallbackConstructorRegistry.constructors = {};
    }
    /**
     * Create callbacks using the registered callback constructors.
     *
     * Given `verbosityLevel`, all constructors registered at that level or above
     * will be called and the instantiated callbacks will be used.
     *
     * @param verbosityLevel: Level of verbosity.
     */
    static createCallbacks(verbosityLevel) {
        const constructors = [];
        for (const levelName in CallbackConstructorRegistry.constructors) {
            const level = +levelName;
            if (verbosityLevel >= level) {
                constructors.push(...CallbackConstructorRegistry.constructors[level]);
            }
        }
        return constructors.map(ctor => new ctor());
    }
}
CallbackConstructorRegistry.constructors = {};
export function configureCallbacks(callbacks, verbose, epochs, initialEpoch, numTrainSamples, stepsPerEpoch, batchSize, doValidation, callbackMetrics) {
    const history = new History();
    const actualCallbacks = [
        new BaseLogger(), ...CallbackConstructorRegistry.createCallbacks(verbose)
    ];
    if (callbacks != null) {
        actualCallbacks.push(...callbacks);
    }
    actualCallbacks.push(history);
    const callbackList = new CallbackList(actualCallbacks);
    // TODO(cais): Figure out when this LayersModel instance can have a
    // dynamically
    //   set property called 'callback_model' as in PyKeras.
    callbackList.setParams({
        epochs,
        initialEpoch,
        samples: numTrainSamples,
        steps: stepsPerEpoch,
        batchSize,
        verbose,
        doValidation,
        metrics: callbackMetrics,
    });
    return { callbackList, history };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFzZV9jYWxsYmFja3MuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvYmFzZV9jYWxsYmFja3MudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFFSCx5Q0FBeUM7QUFFekMsT0FBTyxFQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsSUFBSSxFQUFFLEdBQUcsRUFBRSxTQUFTLEVBQWtCLElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUdqRyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQ3BDLE9BQU8sRUFBTyxvQkFBb0IsRUFBaUIsTUFBTSxRQUFRLENBQUM7QUFDbEUsT0FBTyxLQUFLLGFBQWEsTUFBTSx1QkFBdUIsQ0FBQztBQUV2RCxvREFBb0Q7QUFDcEQsTUFBTSxDQUFOLElBQVkscUJBR1g7QUFIRCxXQUFZLHFCQUFxQjtJQUMvQixxRUFBVSxDQUFBO0lBQ1YsdUVBQVcsQ0FBQTtBQUNiLENBQUMsRUFIVyxxQkFBcUIsS0FBckIscUJBQXFCLFFBR2hDO0FBRUQsbUVBQW1FO0FBQ25FLE1BQU0sQ0FBQyxNQUFNLHNCQUFzQixHQUFHLEdBQUcsQ0FBQztBQVExQzs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQkc7QUFDSCxNQUFNLE9BQWdCLFlBQVk7SUFBbEM7UUFDRSxpREFBaUQ7UUFDakQsbUJBQWMsR0FBb0IsSUFBSSxDQUFDO0lBZ0N6QyxDQUFDO0lBMUJDLFNBQVMsQ0FBQyxNQUFjO1FBQ3RCLElBQUksQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO0lBQ3ZCLENBQUM7SUFFRCxLQUFLLENBQUMsWUFBWSxDQUFDLEtBQWEsRUFBRSxJQUFxQixJQUFHLENBQUM7SUFFM0QsS0FBSyxDQUFDLFVBQVUsQ0FBQyxLQUFhLEVBQUUsSUFBcUIsSUFBRyxDQUFDO0lBRXpELEtBQUssQ0FBQyxZQUFZLENBQUMsS0FBYSxFQUFFLElBQXFCLElBQUcsQ0FBQztJQUUzRCxLQUFLLENBQUMsVUFBVSxDQUFDLEtBQWEsRUFBRSxJQUFxQixJQUFHLENBQUM7SUFFekQsS0FBSyxDQUFDLFlBQVksQ0FBQyxJQUFxQixJQUFHLENBQUM7SUFFNUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxJQUFxQixJQUFHLENBQUM7SUFFMUMsNEVBQTRFO0lBQzVFLDhFQUE4RTtJQUM5RSw4RUFBOEU7SUFDOUUsMEVBQTBFO0lBQzFFLDhFQUE4RTtJQUM5RSxtRUFBbUU7SUFDbkUsZUFBZTtJQUNmLFFBQVEsQ0FBQyxLQUFnQjtRQUN2Qix1RUFBdUU7SUFDekUsQ0FBQztDQUNGO0FBRUQ7O0dBRUc7QUFDSCxNQUFNLE9BQU8sWUFBWTtJQUl2QixzRUFBc0U7SUFDdEUsdUNBQXVDO0lBQ3ZDLCtCQUErQjtJQUMvQiw0Q0FBNEM7SUFDNUMsMENBQTBDO0lBRTFDOzs7OztPQUtHO0lBQ0gsWUFBWSxTQUEwQixFQUFFLFdBQVcsR0FBRyxFQUFFO1FBQ3RELDJFQUEyRTtRQUMzRSxVQUFVO1FBQ1YsSUFBSSxTQUFTLElBQUksSUFBSSxFQUFFO1lBQ3JCLFNBQVMsR0FBRyxFQUFFLENBQUM7U0FDaEI7UUFDRCxJQUFJLENBQUMsU0FBUyxHQUFHLFNBQVMsQ0FBQztRQUMzQixJQUFJLENBQUMsV0FBVyxHQUFHLFdBQVcsQ0FBQztJQUNqQyxDQUFDO0lBRUQsTUFBTSxDQUFDLFFBQXNCO1FBQzNCLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ2hDLENBQUM7SUFFRCxTQUFTLENBQUMsTUFBYztRQUN0QixLQUFLLE1BQU0sUUFBUSxJQUFJLElBQUksQ0FBQyxTQUFTLEVBQUU7WUFDckMsUUFBUSxDQUFDLFNBQVMsQ0FBQyxNQUFNLENBQUMsQ0FBQztTQUM1QjtJQUNILENBQUM7SUFFRCxRQUFRLENBQUMsS0FBZ0I7UUFDdkIsS0FBSyxNQUFNLFFBQVEsSUFBSSxJQUFJLENBQUMsU0FBUyxFQUFFO1lBQ3JDLFFBQVEsQ0FBQyxRQUFRLENBQUMsS0FBSyxDQUFDLENBQUM7U0FDMUI7SUFDSCxDQUFDO0lBRUQ7Ozs7T0FJRztJQUNILEtBQUssQ0FBQyxZQUFZLENBQUMsS0FBYSxFQUFFLElBQXFCO1FBQ3JELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ1g7UUFDRCxLQUFLLE1BQU0sUUFBUSxJQUFJLElBQUksQ0FBQyxTQUFTLEVBQUU7WUFDckMsTUFBTSxRQUFRLENBQUMsWUFBWSxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztTQUMxQztJQUNILENBQUM7SUFFRDs7OztPQUlHO0lBQ0gsS0FBSyxDQUFDLFVBQVUsQ0FBQyxLQUFhLEVBQUUsSUFBcUI7UUFDbkQsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ2hCLElBQUksR0FBRyxFQUFFLENBQUM7U0FDWDtRQUNELEtBQUssTUFBTSxRQUFRLElBQUksSUFBSSxDQUFDLFNBQVMsRUFBRTtZQUNyQyxNQUFNLFFBQVEsQ0FBQyxVQUFVLENBQUMsS0FBSyxFQUFFLElBQUksQ0FBQyxDQUFDO1NBQ3hDO0lBQ0gsQ0FBQztJQUVEOzs7O09BSUc7SUFDSCxLQUFLLENBQUMsWUFBWSxDQUFDLEtBQWEsRUFBRSxJQUFxQjtRQUNyRCxJQUFJLElBQUksSUFBSSxJQUFJLEVBQUU7WUFDaEIsSUFBSSxHQUFHLEVBQUUsQ0FBQztTQUNYO1FBQ0QsS0FBSyxNQUFNLFFBQVEsSUFBSSxJQUFJLENBQUMsU0FBUyxFQUFFO1lBQ3JDLE1BQU0sUUFBUSxDQUFDLFlBQVksQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7U0FDMUM7SUFDSCxDQUFDO0lBRUQ7Ozs7T0FJRztJQUNILEtBQUssQ0FBQyxVQUFVLENBQUMsS0FBYSxFQUFFLElBQXFCO1FBQ25ELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ1g7UUFDRCxLQUFLLE1BQU0sUUFBUSxJQUFJLElBQUksQ0FBQyxTQUFTLEVBQUU7WUFDckMsTUFBTSxRQUFRLENBQUMsVUFBVSxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztTQUN4QztJQUNILENBQUM7SUFFRDs7O09BR0c7SUFDSCxLQUFLLENBQUMsWUFBWSxDQUFDLElBQXFCO1FBQ3RDLElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ1g7UUFDRCxLQUFLLE1BQU0sUUFBUSxJQUFJLElBQUksQ0FBQyxTQUFTLEVBQUU7WUFDckMsTUFBTSxRQUFRLENBQUMsWUFBWSxDQUFDLElBQUksQ0FBQyxDQUFDO1NBQ25DO0lBQ0gsQ0FBQztJQUVEOzs7T0FHRztJQUNILEtBQUssQ0FBQyxVQUFVLENBQUMsSUFBcUI7UUFDcEMsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ2hCLElBQUksR0FBRyxFQUFFLENBQUM7U0FDWDtRQUNELEtBQUssTUFBTSxRQUFRLElBQUksSUFBSSxDQUFDLFNBQVMsRUFBRTtZQUNyQyxNQUFNLFFBQVEsQ0FBQyxVQUFVLENBQUMsSUFBSSxDQUFDLENBQUM7U0FDakM7SUFDSCxDQUFDO0NBQ0Y7QUFFRDs7OztHQUlHO0FBQ0gsTUFBTSxPQUFPLFVBQVcsU0FBUSxZQUFZO0lBSTFDO1FBQ0UsS0FBSyxFQUFFLENBQUM7SUFDVixDQUFDO0lBRVEsS0FBSyxDQUFDLFlBQVksQ0FBQyxLQUFhO1FBQ3ZDLElBQUksQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO1FBQ2QsSUFBSSxDQUFDLE1BQU0sR0FBRyxFQUFFLENBQUM7SUFDbkIsQ0FBQztJQUVRLEtBQUssQ0FBQyxVQUFVLENBQUMsS0FBYSxFQUFFLElBQXFCO1FBQzVELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtZQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDO1NBQ1g7UUFDRCxNQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQVcsQ0FBQztRQUNwRSxJQUFJLENBQUMsSUFBSSxJQUFJLFNBQVMsQ0FBQztRQUN2QixLQUFLLE1BQU0sR0FBRyxJQUFJLElBQUksRUFBRTtZQUN0QixNQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7WUFDeEIsSUFBSSxPQUFPLEtBQUssS0FBSyxRQUFRLEVBQUU7Z0JBQzdCLElBQUksQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxHQUFHLENBQUMsRUFBRTtvQkFDcEMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUM7aUJBQ3RCO2dCQUNELElBQUksQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQVcsR0FBRyxLQUFLLEdBQUcsU0FBUyxDQUFDO2FBQ25FO2lCQUFNO2dCQUNMLElBQUksa0JBQTBCLENBQUM7Z0JBQy9CLElBQUksR0FBRyxJQUFJLElBQUksQ0FBQyxNQUFNLEVBQUU7b0JBQ3RCLGtCQUFrQixHQUFHLElBQUksQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFXLENBQUM7aUJBQ2pEO3FCQUFNO29CQUNMLElBQUksQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO2lCQUN0QjtnQkFDRCxNQUFNLEtBQUssR0FDUCxJQUFJLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMvRCxJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxHQUFHLEtBQUssQ0FBQztnQkFDekIsSUFBSSxrQkFBa0IsSUFBSSxJQUFJLEVBQUU7b0JBQzlCLGtCQUFrQixDQUFDLE9BQU8sRUFBRSxDQUFDO2lCQUM5QjthQUNGO1NBQ0Y7SUFDSCxDQUFDO0lBRVEsS0FBSyxDQUFDLFVBQVUsQ0FBQyxLQUFhLEVBQUUsSUFBcUI7UUFDNUQsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ2hCLEtBQUssTUFBTSxHQUFHLElBQUksSUFBSSxDQUFDLE1BQU0sQ0FBQyxTQUFTLENBQWEsRUFBRTtnQkFDcEQsSUFBSSxJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBQyxJQUFJLElBQUksRUFBRTtvQkFDNUIsU0FBUztpQkFDVjtnQkFDRCxJQUFJLE9BQU8sSUFBSSxDQUFDLE1BQU0sQ0FBQyxHQUFHLENBQUMsS0FBSyxRQUFRLEVBQUU7b0JBQ3hDLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBVyxHQUFHLElBQUksQ0FBQyxJQUFJLENBQUM7aUJBQ3BEO3FCQUFNO29CQUNMLElBQUksQ0FBQyxHQUFHLEVBQUU7d0JBQ1IsTUFBTSxHQUFHLEdBQVcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxFQUFFLElBQUksQ0FBQyxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQzt3QkFDN0QsSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLEdBQUcsQ0FBQzt3QkFDZixJQUFJLENBQUMsTUFBTSxDQUFDLEdBQUcsQ0FBWSxDQUFDLE9BQU8sRUFBRSxDQUFDO3dCQUN2QyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBVyxDQUFDLENBQUM7b0JBQzVCLENBQUMsQ0FBQyxDQUFDO2lCQUNKO2FBQ0Y7U0FDRjtJQUNILENBQUM7Q0FDRjtBQUVEOzs7O0dBSUc7QUFDSCxNQUFNLE9BQU8sT0FBUSxTQUFRLFlBQVk7SUFJOUIsS0FBSyxDQUFDLFlBQVksQ0FBQyxJQUFxQjtRQUMvQyxJQUFJLENBQUMsS0FBSyxHQUFHLEVBQUUsQ0FBQztRQUNoQixJQUFJLENBQUMsT0FBTyxHQUFHLEVBQUUsQ0FBQztJQUNwQixDQUFDO0lBRVEsS0FBSyxDQUFDLFVBQVUsQ0FBQyxLQUFhLEVBQUUsSUFBcUI7UUFDNUQsSUFBSSxJQUFJLElBQUksSUFBSSxFQUFFO1lBQ2hCLElBQUksR0FBRyxFQUFFLENBQUM7U0FDWDtRQUNELElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3ZCLEtBQUssTUFBTSxHQUFHLElBQUksSUFBSSxFQUFFO1lBQ3RCLElBQUksSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsSUFBSSxJQUFJLEVBQUU7Z0JBQzdCLElBQUksQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLEdBQUcsRUFBRSxDQUFDO2FBQ3hCO1lBQ0QsSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUM7U0FDbkM7SUFDSCxDQUFDO0lBRUQ7O09BRUc7SUFDSCxLQUFLLENBQUMsUUFBUTtRQUNaLE1BQU0sUUFBUSxHQUF1RCxFQUFFLENBQUM7UUFDeEUsTUFBTSxJQUFJLEdBQWEsRUFBRSxDQUFDO1FBQzFCLE1BQU0sT0FBTyxHQUFhLEVBQUUsQ0FBQztRQUM3QixLQUFLLE1BQU0sR0FBRyxJQUFJLElBQUksQ0FBQyxPQUFPLEVBQUU7WUFDOUIsTUFBTSxVQUFVLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUNyQyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtnQkFDMUMsSUFBSSxPQUFPLFVBQVUsQ0FBQyxDQUFDLENBQUMsS0FBSyxRQUFRLEVBQUU7b0JBQ3JDLE1BQU0sV0FBVyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQVcsQ0FBQztvQkFDNUMsUUFBUSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQztvQkFDbEMsSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztvQkFDZixPQUFPLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2lCQUNqQjthQUNGO1NBQ0Y7UUFDRCxNQUFNLE1BQU0sR0FBRyxNQUFNLE9BQU8sQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLENBQUM7UUFDM0MsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDdEMsTUFBTSxlQUFlLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQVcsQ0FBQztZQUNwRSxlQUFlLENBQUMsT0FBTyxFQUFFLENBQUM7WUFDMUIsSUFBSSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDbEQ7SUFDSCxDQUFDO0NBQ0Y7QUFlRDs7R0FFRztBQUNILE1BQU0sT0FBTyxjQUFlLFNBQVEsWUFBWTtJQW1COUMsWUFBWSxJQUF3QixFQUFFLFVBQThCO1FBQ2xFLEtBQUssRUFBRSxDQUFDO1FBTEYsaUJBQVksR0FBRyxDQUFDLENBQUM7UUFNdkIsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDO1FBQzVCLElBQUksQ0FBQyxhQUFhLEdBQUcsSUFBSSxDQUFDLGFBQWEsSUFBSSxTQUFTLENBQUM7UUFDckQsSUFBSSxDQUFDLFVBQVUsR0FBRyxVQUFVLElBQUksTUFBTSxDQUFDO1FBQ3ZDLElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxNQUFNLEVBQUU7WUFDOUIsSUFBSSxDQUFDLFVBQVUsR0FBRyxzQkFBc0IsQ0FBQztTQUMxQztRQUNELElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxPQUFPLElBQUksSUFBSSxDQUFDLE9BQU8sSUFBSSxJQUFJLEVBQUU7WUFDdkQsTUFBTSxJQUFJLEtBQUssQ0FDWCxnRUFBZ0U7Z0JBQ2hFLG1EQUFtRCxDQUFDLENBQUM7U0FDMUQ7UUFDRCxJQUFJLElBQUksQ0FBQyxRQUFRLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxFQUFFO1lBQ2xDLCtEQUErRDtZQUMvRCxtQkFBbUI7WUFDbkIsSUFBSSxDQUFDLFNBQVMsR0FBRyxhQUFhLENBQUMsUUFBUSxDQUNuQyxJQUFJLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsRUFBRSxJQUFJLENBQUMsVUFBb0IsRUFBRSxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUM7U0FDekU7UUFDRCxJQUFJLENBQUMsVUFBVSxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUM7UUFDcEMsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUMsVUFBVSxDQUFDO1FBQ2hDLElBQUksQ0FBQyxVQUFVLEdBQUcsSUFBSSxDQUFDLFlBQVksQ0FBQztRQUNwQyxJQUFJLENBQUMsUUFBUSxHQUFHLElBQUksQ0FBQyxVQUFVLENBQUM7UUFDaEMsSUFBSSxDQUFDLFVBQVUsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDO1FBQ3BDLElBQUksQ0FBQyxRQUFRLEdBQUcsSUFBSSxDQUFDLFVBQVUsQ0FBQztRQUNoQyxJQUFJLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQyxPQUFPLENBQUM7SUFDNUIsQ0FBQztJQUVELEtBQUssQ0FBQyxTQUFTLENBQUMsS0FBYSxFQUFFLEtBQWEsRUFBRSxJQUFvQjtRQUNoRSxNQUFNLEVBQUUsR0FBOEIsRUFBRSxDQUFDO1FBQ3pDLElBQUksSUFBSSxDQUFDLEtBQUssSUFBSSxJQUFJLEVBQUU7WUFDdEIsTUFBTSxvQkFBb0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUNqQyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxJQUFZLENBQUMsQ0FBQyxDQUFDO1NBQ2pEO1FBQ0QsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsQ0FBQztRQUM5QixNQUFNLE9BQU8sQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUM7SUFDeEIsQ0FBQztJQUVRLEtBQUssQ0FBQyxZQUFZLENBQUMsS0FBYSxFQUFFLElBQXFCO1FBRTlELElBQUksQ0FBQyxZQUFZLEdBQUcsS0FBSyxDQUFDO1FBQzFCLElBQUksSUFBSSxDQUFDLFVBQVUsSUFBSSxJQUFJLEVBQUU7WUFDM0IsTUFBTSxvQkFBb0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUNqQyxNQUFNLElBQUksQ0FBQyxVQUFVLENBQUMsS0FBSyxFQUFFLElBQVksQ0FBQyxDQUFDO1NBQzVDO0lBQ0gsQ0FBQztJQUVRLEtBQUssQ0FBQyxVQUFVLENBQUMsS0FBYSxFQUFFLElBQXFCO1FBRTVELE1BQU0sRUFBRSxHQUE4QixFQUFFLENBQUM7UUFDekMsSUFBSSxJQUFJLENBQUMsUUFBUSxJQUFJLElBQUksRUFBRTtZQUN6QixNQUFNLG9CQUFvQixDQUFDLElBQUksQ0FBQyxDQUFDO1lBQ2pDLEVBQUUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxLQUFLLEVBQUUsSUFBWSxDQUFDLENBQUMsQ0FBQztTQUM3QztRQUNELElBQUksSUFBSSxDQUFDLFVBQVUsS0FBSyxPQUFPLEVBQUU7WUFDL0IsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsYUFBYSxFQUFFLENBQUMsQ0FBQztTQUMvQjtRQUNELE1BQU0sT0FBTyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUN4QixDQUFDO0lBRVEsS0FBSyxDQUFDLFlBQVksQ0FBQyxLQUFhLEVBQUUsSUFBcUI7UUFFOUQsSUFBSSxJQUFJLENBQUMsVUFBVSxJQUFJLElBQUksRUFBRTtZQUMzQixNQUFNLG9CQUFvQixDQUFDLElBQUksQ0FBQyxDQUFDO1lBQ2pDLE1BQU0sSUFBSSxDQUFDLFVBQVUsQ0FBQyxLQUFLLEVBQUUsSUFBWSxDQUFDLENBQUM7U0FDNUM7SUFDSCxDQUFDO0lBRVEsS0FBSyxDQUFDLFVBQVUsQ0FBQyxLQUFhLEVBQUUsSUFBcUI7UUFFNUQsTUFBTSxFQUFFLEdBQThCLEVBQUUsQ0FBQztRQUN6QyxJQUFJLElBQUksQ0FBQyxRQUFRLElBQUksSUFBSSxFQUFFO1lBQ3pCLE1BQU0sb0JBQW9CLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDakMsRUFBRSxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLEtBQUssRUFBRSxJQUFZLENBQUMsQ0FBQyxDQUFDO1NBQzdDO1FBQ0QsSUFBSSxJQUFJLENBQUMsVUFBVSxLQUFLLE9BQU8sRUFBRTtZQUMvQixFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxhQUFhLEVBQUUsQ0FBQyxDQUFDO1NBQy9CO2FBQU0sSUFBSSxJQUFJLENBQUMsUUFBUSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRTtZQUN6QyxFQUFFLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsSUFBSSxDQUFDLFlBQVksRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztTQUN6RDtRQUNELE1BQU0sT0FBTyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUN4QixDQUFDO0lBRVEsS0FBSyxDQUFDLFlBQVksQ0FBQyxJQUFxQjtRQUMvQyxJQUFJLElBQUksQ0FBQyxVQUFVLElBQUksSUFBSSxFQUFFO1lBQzNCLE1BQU0sb0JBQW9CLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDakMsTUFBTSxJQUFJLENBQUMsVUFBVSxDQUFDLElBQVksQ0FBQyxDQUFDO1NBQ3JDO0lBQ0gsQ0FBQztJQUVRLEtBQUssQ0FBQyxVQUFVLENBQUMsSUFBcUI7UUFDN0MsSUFBSSxJQUFJLENBQUMsUUFBUSxJQUFJLElBQUksRUFBRTtZQUN6QixNQUFNLG9CQUFvQixDQUFDLElBQUksQ0FBQyxDQUFDO1lBQ2pDLE1BQU0sSUFBSSxDQUFDLFFBQVEsQ0FBQyxJQUFZLENBQUMsQ0FBQztTQUNuQztJQUNILENBQUM7Q0FDRjtBQUVEOztHQUVHO0FBQ0gsTUFBTSxVQUFVLG9CQUFvQixDQUNoQyxTQUNvQixFQUNwQixVQUE2QjtJQUMvQixJQUFJLFNBQVMsSUFBSSxJQUFJLEVBQUU7UUFDckIsU0FBUyxHQUFHLEVBQWtCLENBQUM7S0FDaEM7SUFDRCxJQUFJLFNBQVMsWUFBWSxZQUFZLEVBQUU7UUFDckMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO0tBQ3BCO0lBQ0QsSUFBSSxLQUFLLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxJQUFJLFNBQVMsQ0FBQyxDQUFDLENBQUMsWUFBWSxZQUFZLEVBQUU7UUFDcEUsT0FBTyxTQUEyQixDQUFDO0tBQ3BDO0lBQ0QsOERBQThEO0lBQzlELE1BQU0sZUFBZSxHQUNqQixhQUFhLENBQUMsTUFBTSxDQUFDLFNBQVMsQ0FBeUIsQ0FBQztJQUM1RCxPQUFPLGVBQWUsQ0FBQyxHQUFHLENBQ3RCLGNBQWMsQ0FBQyxFQUFFLENBQUMsSUFBSSxjQUFjLENBQUMsY0FBYyxFQUFFLFVBQVUsQ0FBQyxDQUFDLENBQUM7QUFDeEUsQ0FBQztBQU1EOzs7R0FHRztBQUNILE1BQU0sT0FBTywyQkFBMkI7SUFJdEM7O09BRUc7SUFDSCxnQkFBdUIsQ0FBQztJQUV4Qjs7Ozs7Ozs7Ozs7T0FXRztJQUNILE1BQU0sQ0FBQywyQkFBMkIsQ0FDOUIsY0FBc0IsRUFBRSxtQkFBNEM7UUFDdEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxjQUFjLElBQUksQ0FBQyxJQUFJLE1BQU0sQ0FBQyxTQUFTLENBQUMsY0FBYyxDQUFDLEVBQ3ZELEdBQUcsRUFBRSxDQUFDLHFEQUFxRDtZQUN2RCxXQUFXLGNBQWMsRUFBRSxDQUFDLENBQUM7UUFDckMsMkJBQTJCLENBQUMsaUJBQWlCLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUNuRSxJQUFJLDJCQUEyQixDQUFDLFlBQVksQ0FBQyxjQUFjLENBQUMsSUFBSSxJQUFJLEVBQUU7WUFDcEUsMkJBQTJCLENBQUMsWUFBWSxDQUFDLGNBQWMsQ0FBQyxHQUFHLEVBQUUsQ0FBQztTQUMvRDtRQUNELDJCQUEyQixDQUFDLFlBQVksQ0FBQyxjQUFjLENBQUMsQ0FBQyxJQUFJLENBQ3pELG1CQUFtQixDQUFDLENBQUM7SUFDM0IsQ0FBQztJQUVPLE1BQU0sQ0FBQyxpQkFBaUIsQ0FBQyxtQkFDMkI7UUFDMUQsS0FBSyxNQUFNLFNBQVMsSUFBSSwyQkFBMkIsQ0FBQyxZQUFZLEVBQUU7WUFDaEUsTUFBTSxZQUFZLEdBQUcsMkJBQTJCLENBQUMsWUFBWSxDQUFDLENBQUMsU0FBUyxDQUFDLENBQUM7WUFDMUUsWUFBWSxDQUFDLE9BQU8sQ0FBQyxJQUFJLENBQUMsRUFBRTtnQkFDMUIsSUFBSSxJQUFJLEtBQUssbUJBQW1CLEVBQUU7b0JBQ2hDLE1BQU0sSUFBSSxVQUFVLENBQUMsaUNBQWlDLENBQUMsQ0FBQztpQkFDekQ7WUFDSCxDQUFDLENBQUMsQ0FBQztTQUNKO0lBQ0gsQ0FBQztJQUVEOztPQUVHO0lBQ08sTUFBTSxDQUFDLEtBQUs7UUFDcEIsMkJBQTJCLENBQUMsWUFBWSxHQUFHLEVBQUUsQ0FBQztJQUNoRCxDQUFDO0lBRUQ7Ozs7Ozs7T0FPRztJQUNILE1BQU0sQ0FBQyxlQUFlLENBQUMsY0FBc0I7UUFDM0MsTUFBTSxZQUFZLEdBQThCLEVBQUUsQ0FBQztRQUNuRCxLQUFLLE1BQU0sU0FBUyxJQUFJLDJCQUEyQixDQUFDLFlBQVksRUFBRTtZQUNoRSxNQUFNLEtBQUssR0FBRyxDQUFDLFNBQVMsQ0FBQztZQUN6QixJQUFJLGNBQWMsSUFBSSxLQUFLLEVBQUU7Z0JBQzNCLFlBQVksQ0FBQyxJQUFJLENBQUMsR0FBRywyQkFBMkIsQ0FBQyxZQUFZLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQzthQUN2RTtTQUNGO1FBQ0QsT0FBTyxZQUFZLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsSUFBSSxJQUFJLEVBQUUsQ0FBQyxDQUFDO0lBQzlDLENBQUM7O0FBdEVjLHdDQUFZLEdBQ2lDLEVBQUUsQ0FBQztBQXdFakUsTUFBTSxVQUFVLGtCQUFrQixDQUM5QixTQUF5QixFQUFFLE9BQThCLEVBQUUsTUFBYyxFQUN6RSxZQUFvQixFQUFFLGVBQXVCLEVBQUUsYUFBcUIsRUFDcEUsU0FBaUIsRUFBRSxZQUFxQixFQUN4QyxlQUF5QjtJQUMzQixNQUFNLE9BQU8sR0FBRyxJQUFJLE9BQU8sRUFBRSxDQUFDO0lBQzlCLE1BQU0sZUFBZSxHQUFtQjtRQUN0QyxJQUFJLFVBQVUsRUFBRSxFQUFFLEdBQUcsMkJBQTJCLENBQUMsZUFBZSxDQUFDLE9BQU8sQ0FBQztLQUMxRSxDQUFDO0lBQ0YsSUFBSSxTQUFTLElBQUksSUFBSSxFQUFFO1FBQ3JCLGVBQWUsQ0FBQyxJQUFJLENBQUMsR0FBRyxTQUFTLENBQUMsQ0FBQztLQUNwQztJQUNELGVBQWUsQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDOUIsTUFBTSxZQUFZLEdBQUcsSUFBSSxZQUFZLENBQUMsZUFBZSxDQUFDLENBQUM7SUFFdkQsbUVBQW1FO0lBQ25FLGNBQWM7SUFDZCx3REFBd0Q7SUFFeEQsWUFBWSxDQUFDLFNBQVMsQ0FBQztRQUNyQixNQUFNO1FBQ04sWUFBWTtRQUNaLE9BQU8sRUFBRSxlQUFlO1FBQ3hCLEtBQUssRUFBRSxhQUFhO1FBQ3BCLFNBQVM7UUFDVCxPQUFPO1FBQ1AsWUFBWTtRQUNaLE9BQU8sRUFBRSxlQUFlO0tBQ3pCLENBQUMsQ0FBQztJQUNILE9BQU8sRUFBQyxZQUFZLEVBQUUsT0FBTyxFQUFDLENBQUM7QUFDakMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qIE9yaWdpbmFsIHNvdXJjZToga2VyYXMvY2FsbGJhY2tzLnB5ICovXG5cbmltcG9ydCB7YWRkLCBkaXYsIGtlZXAsIG11bCwgbmV4dEZyYW1lLCBTY2FsYXIsIFRlbnNvciwgdGlkeSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtDb250YWluZXJ9IGZyb20gJy4vZW5naW5lL2NvbnRhaW5lcic7XG5pbXBvcnQge1ZhbHVlRXJyb3J9IGZyb20gJy4vZXJyb3JzJztcbmltcG9ydCB7TG9ncywgcmVzb2x2ZVNjYWxhcnNJbkxvZ3MsIFVucmVzb2x2ZWRMb2dzfSBmcm9tICcuL2xvZ3MnO1xuaW1wb3J0ICogYXMgZ2VuZXJpY191dGlscyBmcm9tICcuL3V0aWxzL2dlbmVyaWNfdXRpbHMnO1xuXG4vKiogVmVyYm9zaXR5IGxvZ2dpbmcgbGV2ZWwgd2hlbiBmaXR0aW5nIGEgbW9kZWwuICovXG5leHBvcnQgZW51bSBNb2RlbExvZ2dpbmdWZXJib3NpdHkge1xuICBTSUxFTlQgPSAwLFxuICBWRVJCT1NFID0gMVxufVxuXG4vKiogSG93IG9mdGVuIHRvIHlpZWxkIHRvIHRoZSBtYWluIHRocmVhZCB3aGVuIHRyYWluaW5nIChpbiBtcykuICovXG5leHBvcnQgY29uc3QgREVGQVVMVF9ZSUVMRF9FVkVSWV9NUyA9IDEyNTtcblxuZXhwb3J0IHR5cGUgUGFyYW1zID0ge1xuICBba2V5OiBzdHJpbmddOiBudW1iZXJ8c3RyaW5nfGJvb2xlYW58bnVtYmVyW118c3RyaW5nW118Ym9vbGVhbltdO1xufTtcblxuZXhwb3J0IHR5cGUgWWllbGRFdmVyeU9wdGlvbnMgPSAnYXV0byd8J2JhdGNoJ3wnZXBvY2gnfCduZXZlcid8bnVtYmVyO1xuXG4vKipcbiAqIEFic3RyYWN0IGJhc2UgY2xhc3MgdXNlZCB0byBidWlsZCBuZXcgY2FsbGJhY2tzLlxuICpcbiAqIFRoZSBgbG9nc2AgZGljdGlvbmFyeSB0aGF0IGNhbGxiYWNrIG1ldGhvZHMgdGFrZSBhcyBhcmd1bWVudCB3aWxsIGNvbnRhaW5cbiAqIGtleXMgZm9yIHF1YW50aXRpZXMgcmVsZXZhbnQgdG8gdGhlIGN1cnJlbnQgYmF0Y2ggb3IgZXBvY2guXG4gKlxuICogQ3VycmVudGx5LCB0aGUgYC5maXQoKWAgbWV0aG9kIG9mIHRoZSBgU2VxdWVudGlhbGAgbW9kZWwgY2xhc3NcbiAqIHdpbGwgaW5jbHVkZSB0aGUgZm9sbG93aW5nIHF1YW50aXRpZXMgaW4gdGhlIGBsb2dzYCB0aGF0XG4gKiBpdCBwYXNzZXMgdG8gaXRzIGNhbGxiYWNrczpcbiAqXG4gKiBvbkVwb2NoRW5kOiBMb2dzIGluY2x1ZGUgYGFjY2AgYW5kIGBsb3NzYCwgYW5kIG9wdGlvbmFsbHkgaW5jbHVkZSBgdmFsTG9zc2BcbiAqICAgKGlmIHZhbGlkYXRpb24gaXMgZW5hYmxlZCBpbiBgZml0YCksIGFuZCBgdmFsQWNjYCAoaWYgdmFsaWRhdGlvbiBhbmRcbiAqICAgYWNjdXJhY3kgbW9uaXRvcmluZyBhcmUgZW5hYmxlZCkuXG4gKiBvbkJhdGNoQmVnaW46IExvZ3MgaW5jbHVkZSBgc2l6ZWAsIHRoZSBudW1iZXIgb2Ygc2FtcGxlcyBpbiB0aGUgY3VycmVudFxuICogICBiYXRjaC5cbiAqIG9uQmF0Y2hFbmQ6IExvZ3MgaW5jbHVkZSBgbG9zc2AsIGFuZCBvcHRpb25hbGx5IGBhY2NgIChpZiBhY2N1cmFjeSBtb25pdG9yaW5nXG4gKiAgIGlzIGVuYWJsZWQpLlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgQmFzZUNhbGxiYWNrIHtcbiAgLy8gVE9ETyhtaWNoYWVsdGVycnkpOiBUaGlzIHR5cGUgaXMgYSBiZXN0IGd1ZXNzLlxuICB2YWxpZGF0aW9uRGF0YTogVGVuc29yfFRlbnNvcltdID0gbnVsbDtcbiAgLyoqXG4gICAqIFRyYWluaW5nIHBhcmFtZXRlcnMgKGVnLiB2ZXJib3NpdHksIGJhdGNoIHNpemUsIG51bWJlciBvZiBlcG9jaHMuLi4pLlxuICAgKi9cbiAgcGFyYW1zOiBQYXJhbXM7XG5cbiAgc2V0UGFyYW1zKHBhcmFtczogUGFyYW1zKTogdm9pZCB7XG4gICAgdGhpcy5wYXJhbXMgPSBwYXJhbXM7XG4gIH1cblxuICBhc3luYyBvbkVwb2NoQmVnaW4oZXBvY2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7fVxuXG4gIGFzeW5jIG9uRXBvY2hFbmQoZXBvY2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7fVxuXG4gIGFzeW5jIG9uQmF0Y2hCZWdpbihiYXRjaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHt9XG5cbiAgYXN5bmMgb25CYXRjaEVuZChiYXRjaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHt9XG5cbiAgYXN5bmMgb25UcmFpbkJlZ2luKGxvZ3M/OiBVbnJlc29sdmVkTG9ncykge31cblxuICBhc3luYyBvblRyYWluRW5kKGxvZ3M/OiBVbnJlc29sdmVkTG9ncykge31cblxuICAvLyBMYXllcnNNb2RlbCBuZWVkcyB0byBjYWxsIENhbGxiYWNrLnNldE1vZGVsKCksIGJ1dCBjYW5ub3QgYWN0dWFsbHkgZGVwZW5kXG4gIC8vIG9uIENhbGxiYWNrIGJlY2F1c2UgdGhhdCBjcmVhdGVzIGEgY3ljbGljIGRlcGVuZGVuY3kuICBQcm92aWRpbmcgdGhpcyBuby1vcFxuICAvLyBtZXRob2Qgb24gQmFzZUNhbGxiYWNrIGJyZWFrcyB0aGUgY3ljbGU6IHRoaXMgd2F5IExheWVyc01vZGVsIGNhbiBkZXBlbmQgb25cbiAgLy8gQmFzZUNhbGxiYWNrIGJ1dCBub3Qgb24gQ2FsbGJhY2suICBUaGUgYXJndW1lbnQgaXMgdHlwZWQgYXMgYENvbnRhaW5lcmBcbiAgLy8gKHRoZSBzdXBlcmNsYXNzIG9mIExheWVyc01vZGVsKSB0byBhdm9pZCByZWNhcGl0dWxhdGluZyB0aGUgY3ljbGUuIENhbGxiYWNrXG4gIC8vIG92ZXJyaWRlcyB0aGlzIG1ldGhvZCBhbmQgZW5mb3JjZXMgdGhhdCB0aGUgYXJndW1lbnQgaXMgcmVhbGx5IGFcbiAgLy8gTGF5ZXJzTW9kZWwuXG4gIHNldE1vZGVsKG1vZGVsOiBDb250YWluZXIpOiB2b2lkIHtcbiAgICAvLyBEbyBub3RoaW5nLiBVc2UgQ2FsbGJhY2sgaW5zdGVhZCBvZiBCYXNlQ2FsbGJhY2sgdG8gdHJhY2sgdGhlIG1vZGVsLlxuICB9XG59XG5cbi8qKlxuICogQ29udGFpbmVyIGFic3RyYWN0aW5nIGEgbGlzdCBvZiBjYWxsYmFja3MuXG4gKi9cbmV4cG9ydCBjbGFzcyBDYWxsYmFja0xpc3Qge1xuICBjYWxsYmFja3M6IEJhc2VDYWxsYmFja1tdO1xuICBxdWV1ZUxlbmd0aDogbnVtYmVyO1xuXG4gIC8vIFRPRE8oY2Fpcyk6IFdoZW4gdGhlIG5lZWQgYXJpc2VzLCB1bmNvbW1lbnQgdGhlIGZvbGxvd2luZyBsaW5lcyBhbmRcbiAgLy8gaW1wbGVtZW50IHRoZSBxdWV1ZSBmb3IgdGltZSB2YWx1ZXMuXG4gIC8vIHByaXZhdGUgZGVsdGFUQmF0Y2g6IG51bWJlcjtcbiAgLy8gcHJpdmF0ZSBkZWx0YVRzQmF0Y2hCZWdpbjogQXJyYXk8bnVtYmVyPjtcbiAgLy8gcHJpdmF0ZSBkZWx0YVRzQmF0Y2hFbmQ6IEFycmF5PG51bWJlcj47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdG9yIG9mIENhbGxiYWNrTGlzdC5cbiAgICogQHBhcmFtIGNhbGxiYWNrcyBBcnJheSBvZiBgQ2FsbGJhY2tgIGluc3RhbmNlcy5cbiAgICogQHBhcmFtIHF1ZXVlTGVuZ3RoIFF1ZXVlIGxlbmd0aCBmb3Iga2VlcGluZyBydW5uaW5nIHN0YXRpc3RpY3Mgb3ZlclxuICAgKiAgIGNhbGxiYWNrIGV4ZWN1dGlvbiB0aW1lLlxuICAgKi9cbiAgY29uc3RydWN0b3IoY2FsbGJhY2tzPzogQmFzZUNhbGxiYWNrW10sIHF1ZXVlTGVuZ3RoID0gMTApIHtcbiAgICAvLyBUT0RPKGNhaXMpOiBNYWtlIHVzZSBvZiBxdWV1ZUxlbmd0aCB3aGVuIGltcGxlbWVudGluZyB0aGUgcXVldWUgZm9yIHRpbWVcbiAgICAvLyB2YWx1ZXMuXG4gICAgaWYgKGNhbGxiYWNrcyA9PSBudWxsKSB7XG4gICAgICBjYWxsYmFja3MgPSBbXTtcbiAgICB9XG4gICAgdGhpcy5jYWxsYmFja3MgPSBjYWxsYmFja3M7XG4gICAgdGhpcy5xdWV1ZUxlbmd0aCA9IHF1ZXVlTGVuZ3RoO1xuICB9XG5cbiAgYXBwZW5kKGNhbGxiYWNrOiBCYXNlQ2FsbGJhY2spOiB2b2lkIHtcbiAgICB0aGlzLmNhbGxiYWNrcy5wdXNoKGNhbGxiYWNrKTtcbiAgfVxuXG4gIHNldFBhcmFtcyhwYXJhbXM6IFBhcmFtcyk6IHZvaWQge1xuICAgIGZvciAoY29uc3QgY2FsbGJhY2sgb2YgdGhpcy5jYWxsYmFja3MpIHtcbiAgICAgIGNhbGxiYWNrLnNldFBhcmFtcyhwYXJhbXMpO1xuICAgIH1cbiAgfVxuXG4gIHNldE1vZGVsKG1vZGVsOiBDb250YWluZXIpOiB2b2lkIHtcbiAgICBmb3IgKGNvbnN0IGNhbGxiYWNrIG9mIHRoaXMuY2FsbGJhY2tzKSB7XG4gICAgICBjYWxsYmFjay5zZXRNb2RlbChtb2RlbCk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIENhbGxlZCBhdCB0aGUgc3RhcnQgb2YgYW4gZXBvY2guXG4gICAqIEBwYXJhbSBlcG9jaCBJbmRleCBvZiBlcG9jaC5cbiAgICogQHBhcmFtIGxvZ3MgRGljdGlvbmFyeSBvZiBsb2dzLlxuICAgKi9cbiAgYXN5bmMgb25FcG9jaEJlZ2luKGVwb2NoOiBudW1iZXIsIGxvZ3M/OiBVbnJlc29sdmVkTG9ncykge1xuICAgIGlmIChsb2dzID09IG51bGwpIHtcbiAgICAgIGxvZ3MgPSB7fTtcbiAgICB9XG4gICAgZm9yIChjb25zdCBjYWxsYmFjayBvZiB0aGlzLmNhbGxiYWNrcykge1xuICAgICAgYXdhaXQgY2FsbGJhY2sub25FcG9jaEJlZ2luKGVwb2NoLCBsb2dzKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogQ2FsbGVkIGF0IHRoZSBlbmQgb2YgYW4gZXBvY2guXG4gICAqIEBwYXJhbSBlcG9jaCBJbmRleCBvZiBlcG9jaC5cbiAgICogQHBhcmFtIGxvZ3MgRGljdGlvbmFyeSBvZiBsb2dzLlxuICAgKi9cbiAgYXN5bmMgb25FcG9jaEVuZChlcG9jaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHtcbiAgICBpZiAobG9ncyA9PSBudWxsKSB7XG4gICAgICBsb2dzID0ge307XG4gICAgfVxuICAgIGZvciAoY29uc3QgY2FsbGJhY2sgb2YgdGhpcy5jYWxsYmFja3MpIHtcbiAgICAgIGF3YWl0IGNhbGxiYWNrLm9uRXBvY2hFbmQoZXBvY2gsIGxvZ3MpO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBDYWxsZWQgIHJpZ2h0IGJlZm9yZSBwcm9jZXNzaW5nIGEgYmF0Y2guXG4gICAqIEBwYXJhbSBiYXRjaCBJbmRleCBvZiBiYXRjaCB3aXRoaW4gdGhlIGN1cnJlbnQgZXBvY2guXG4gICAqIEBwYXJhbSBsb2dzIERpY3Rpb25hcnkgb2YgbG9ncy5cbiAgICovXG4gIGFzeW5jIG9uQmF0Y2hCZWdpbihiYXRjaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHtcbiAgICBpZiAobG9ncyA9PSBudWxsKSB7XG4gICAgICBsb2dzID0ge307XG4gICAgfVxuICAgIGZvciAoY29uc3QgY2FsbGJhY2sgb2YgdGhpcy5jYWxsYmFja3MpIHtcbiAgICAgIGF3YWl0IGNhbGxiYWNrLm9uQmF0Y2hCZWdpbihiYXRjaCwgbG9ncyk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIENhbGxlZCBhdCB0aGUgZW5kIG9mIGEgYmF0Y2guXG4gICAqIEBwYXJhbSBiYXRjaCBJbmRleCBvZiBiYXRjaCB3aXRoaW4gdGhlIGN1cnJlbnQgZXBvY2guXG4gICAqIEBwYXJhbSBsb2dzIERpY3Rpb25hcnkgb2YgbG9ncy5cbiAgICovXG4gIGFzeW5jIG9uQmF0Y2hFbmQoYmF0Y2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7XG4gICAgaWYgKGxvZ3MgPT0gbnVsbCkge1xuICAgICAgbG9ncyA9IHt9O1xuICAgIH1cbiAgICBmb3IgKGNvbnN0IGNhbGxiYWNrIG9mIHRoaXMuY2FsbGJhY2tzKSB7XG4gICAgICBhd2FpdCBjYWxsYmFjay5vbkJhdGNoRW5kKGJhdGNoLCBsb2dzKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogQ2FsbGVkIGF0IHRoZSBiZWdpbm5pbmcgb2YgdHJhaW5pbmcuXG4gICAqIEBwYXJhbSBsb2dzIERpY3Rpb25hcnkgb2YgbG9ncy5cbiAgICovXG4gIGFzeW5jIG9uVHJhaW5CZWdpbihsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHtcbiAgICBpZiAobG9ncyA9PSBudWxsKSB7XG4gICAgICBsb2dzID0ge307XG4gICAgfVxuICAgIGZvciAoY29uc3QgY2FsbGJhY2sgb2YgdGhpcy5jYWxsYmFja3MpIHtcbiAgICAgIGF3YWl0IGNhbGxiYWNrLm9uVHJhaW5CZWdpbihsb2dzKTtcbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogQ2FsbGVkIGF0IHRoZSBlbmQgb2YgdHJhaW5pbmcuXG4gICAqIEBwYXJhbSBsb2dzIERpY3Rpb25hcnkgb2YgbG9ncy5cbiAgICovXG4gIGFzeW5jIG9uVHJhaW5FbmQobG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7XG4gICAgaWYgKGxvZ3MgPT0gbnVsbCkge1xuICAgICAgbG9ncyA9IHt9O1xuICAgIH1cbiAgICBmb3IgKGNvbnN0IGNhbGxiYWNrIG9mIHRoaXMuY2FsbGJhY2tzKSB7XG4gICAgICBhd2FpdCBjYWxsYmFjay5vblRyYWluRW5kKGxvZ3MpO1xuICAgIH1cbiAgfVxufVxuXG4vKipcbiAqIENhbGxiYWNrIHRoYXQgYWNjdW11bGF0ZXMgZXBvY2ggYXZlcmFnZXMgb2YgbWV0cmljcy5cbiAqXG4gKiBUaGlzIGNhbGxiYWNrIGlzIGF1dG9tYXRpY2FsbHkgYXBwbGllZCB0byBldmVyeSBMYXllcnNNb2RlbC5cbiAqL1xuZXhwb3J0IGNsYXNzIEJhc2VMb2dnZXIgZXh0ZW5kcyBCYXNlQ2FsbGJhY2sge1xuICBwcml2YXRlIHNlZW46IG51bWJlcjtcbiAgcHJpdmF0ZSB0b3RhbHM6IFVucmVzb2x2ZWRMb2dzO1xuXG4gIGNvbnN0cnVjdG9yKCkge1xuICAgIHN1cGVyKCk7XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyBvbkVwb2NoQmVnaW4oZXBvY2g6IG51bWJlcikge1xuICAgIHRoaXMuc2VlbiA9IDA7XG4gICAgdGhpcy50b3RhbHMgPSB7fTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIG9uQmF0Y2hFbmQoYmF0Y2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7XG4gICAgaWYgKGxvZ3MgPT0gbnVsbCkge1xuICAgICAgbG9ncyA9IHt9O1xuICAgIH1cbiAgICBjb25zdCBiYXRjaFNpemUgPSBsb2dzWydzaXplJ10gPT0gbnVsbCA/IDAgOiBsb2dzWydzaXplJ10gYXMgbnVtYmVyO1xuICAgIHRoaXMuc2VlbiArPSBiYXRjaFNpemU7XG4gICAgZm9yIChjb25zdCBrZXkgaW4gbG9ncykge1xuICAgICAgY29uc3QgdmFsdWUgPSBsb2dzW2tleV07XG4gICAgICBpZiAodHlwZW9mIHZhbHVlID09PSAnbnVtYmVyJykge1xuICAgICAgICBpZiAoIXRoaXMudG90YWxzLmhhc093blByb3BlcnR5KGtleSkpIHtcbiAgICAgICAgICB0aGlzLnRvdGFsc1trZXldID0gMDtcbiAgICAgICAgfVxuICAgICAgICB0aGlzLnRvdGFsc1trZXldID0gdGhpcy50b3RhbHNba2V5XSBhcyBudW1iZXIgKyB2YWx1ZSAqIGJhdGNoU2l6ZTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGxldCBvbGRUb3RhbHNUb0Rpc3Bvc2U6IFNjYWxhcjtcbiAgICAgICAgaWYgKGtleSBpbiB0aGlzLnRvdGFscykge1xuICAgICAgICAgIG9sZFRvdGFsc1RvRGlzcG9zZSA9IHRoaXMudG90YWxzW2tleV0gYXMgU2NhbGFyO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRoaXMudG90YWxzW2tleV0gPSAwO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IHRvdGFsOiBTY2FsYXIgPVxuICAgICAgICAgICAgdGlkeSgoKSA9PiBhZGQoKHRoaXMudG90YWxzW2tleV0pLCBtdWwodmFsdWUsIGJhdGNoU2l6ZSkpKTtcbiAgICAgICAgdGhpcy50b3RhbHNba2V5XSA9IHRvdGFsO1xuICAgICAgICBpZiAob2xkVG90YWxzVG9EaXNwb3NlICE9IG51bGwpIHtcbiAgICAgICAgICBvbGRUb3RhbHNUb0Rpc3Bvc2UuZGlzcG9zZSgpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgb25FcG9jaEVuZChlcG9jaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpIHtcbiAgICBpZiAobG9ncyAhPSBudWxsKSB7XG4gICAgICBmb3IgKGNvbnN0IGtleSBvZiB0aGlzLnBhcmFtc1snbWV0cmljcyddIGFzIHN0cmluZ1tdKSB7XG4gICAgICAgIGlmICh0aGlzLnRvdGFsc1trZXldID09IG51bGwpIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICBpZiAodHlwZW9mIHRoaXMudG90YWxzW2tleV0gPT09ICdudW1iZXInKSB7XG4gICAgICAgICAgbG9nc1trZXldID0gdGhpcy50b3RhbHNba2V5XSBhcyBudW1iZXIgLyB0aGlzLnNlZW47XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgdGlkeSgoKSA9PiB7XG4gICAgICAgICAgICBjb25zdCBsb2c6IFNjYWxhciA9IG11bChkaXYoMSwgdGhpcy5zZWVuKSwgdGhpcy50b3RhbHNba2V5XSk7XG4gICAgICAgICAgICBsb2dzW2tleV0gPSBsb2c7XG4gICAgICAgICAgICAodGhpcy50b3RhbHNba2V5XSBhcyBUZW5zb3IpLmRpc3Bvc2UoKTtcbiAgICAgICAgICAgIGtlZXAobG9nc1trZXldIGFzIFNjYWxhcik7XG4gICAgICAgICAgfSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cblxuLyoqXG4gKiBDYWxsYmFjayB0aGF0IHJlY29yZHMgZXZlbnRzIGludG8gYSBgSGlzdG9yeWAgb2JqZWN0LiBUaGlzIGNhbGxiYWNrIGlzXG4gKiBhdXRvbWF0aWNhbGx5IGFwcGxpZWQgdG8gZXZlcnkgVEYuanMgTGF5ZXJzIG1vZGVsLiBUaGUgYEhpc3RvcnlgIG9iamVjdFxuICogZ2V0cyByZXR1cm5lZCBieSB0aGUgYGZpdGAgbWV0aG9kIG9mIG1vZGVscy5cbiAqL1xuZXhwb3J0IGNsYXNzIEhpc3RvcnkgZXh0ZW5kcyBCYXNlQ2FsbGJhY2sge1xuICBlcG9jaDogbnVtYmVyW107XG4gIGhpc3Rvcnk6IHtba2V5OiBzdHJpbmddOiBBcnJheTxudW1iZXJ8VGVuc29yPn07XG5cbiAgb3ZlcnJpZGUgYXN5bmMgb25UcmFpbkJlZ2luKGxvZ3M/OiBVbnJlc29sdmVkTG9ncykge1xuICAgIHRoaXMuZXBvY2ggPSBbXTtcbiAgICB0aGlzLmhpc3RvcnkgPSB7fTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIG9uRXBvY2hFbmQoZXBvY2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKSB7XG4gICAgaWYgKGxvZ3MgPT0gbnVsbCkge1xuICAgICAgbG9ncyA9IHt9O1xuICAgIH1cbiAgICB0aGlzLmVwb2NoLnB1c2goZXBvY2gpO1xuICAgIGZvciAoY29uc3Qga2V5IGluIGxvZ3MpIHtcbiAgICAgIGlmICh0aGlzLmhpc3Rvcnlba2V5XSA9PSBudWxsKSB7XG4gICAgICAgIHRoaXMuaGlzdG9yeVtrZXldID0gW107XG4gICAgICB9XG4gICAgICB0aGlzLmhpc3Rvcnlba2V5XS5wdXNoKGxvZ3Nba2V5XSk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIEF3YWl0IHRoZSB2YWx1ZXMgb2YgYWxsIGxvc3NlcyBhbmQgbWV0cmljcy5cbiAgICovXG4gIGFzeW5jIHN5bmNEYXRhKCkge1xuICAgIGNvbnN0IHByb21pc2VzOiBBcnJheTxQcm9taXNlPEZsb2F0MzJBcnJheXxJbnQzMkFycmF5fFVpbnQ4QXJyYXk+PiA9IFtdO1xuICAgIGNvbnN0IGtleXM6IHN0cmluZ1tdID0gW107XG4gICAgY29uc3QgaW5kaWNlczogbnVtYmVyW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IGtleSBpbiB0aGlzLmhpc3RvcnkpIHtcbiAgICAgIGNvbnN0IHZhbHVlQXJyYXkgPSB0aGlzLmhpc3Rvcnlba2V5XTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVBcnJheS5sZW5ndGg7ICsraSkge1xuICAgICAgICBpZiAodHlwZW9mIHZhbHVlQXJyYXlbaV0gIT09ICdudW1iZXInKSB7XG4gICAgICAgICAgY29uc3QgdmFsdWVTY2FsYXIgPSB2YWx1ZUFycmF5W2ldIGFzIFRlbnNvcjtcbiAgICAgICAgICBwcm9taXNlcy5wdXNoKHZhbHVlU2NhbGFyLmRhdGEoKSk7XG4gICAgICAgICAga2V5cy5wdXNoKGtleSk7XG4gICAgICAgICAgaW5kaWNlcy5wdXNoKGkpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIGNvbnN0IHZhbHVlcyA9IGF3YWl0IFByb21pc2UuYWxsKHByb21pc2VzKTtcbiAgICBmb3IgKGxldCBuID0gMDsgbiA8IHZhbHVlcy5sZW5ndGg7ICsrbikge1xuICAgICAgY29uc3QgdGVuc29yVG9EaXNwb3NlID0gdGhpcy5oaXN0b3J5W2tleXNbbl1dW2luZGljZXNbbl1dIGFzIFRlbnNvcjtcbiAgICAgIHRlbnNvclRvRGlzcG9zZS5kaXNwb3NlKCk7XG4gICAgICB0aGlzLmhpc3Rvcnlba2V5c1tuXV1baW5kaWNlc1tuXV0gPSB2YWx1ZXNbbl1bMF07XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgQ3VzdG9tQ2FsbGJhY2tBcmdzIHtcbiAgb25UcmFpbkJlZ2luPzogKGxvZ3M/OiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgb25UcmFpbkVuZD86IChsb2dzPzogTG9ncykgPT4gdm9pZCB8IFByb21pc2U8dm9pZD47XG4gIG9uRXBvY2hCZWdpbj86IChlcG9jaDogbnVtYmVyLCBsb2dzPzogTG9ncykgPT4gdm9pZCB8IFByb21pc2U8dm9pZD47XG4gIG9uRXBvY2hFbmQ/OiAoZXBvY2g6IG51bWJlciwgbG9ncz86IExvZ3MpID0+IHZvaWQgfCBQcm9taXNlPHZvaWQ+O1xuICBvbkJhdGNoQmVnaW4/OiAoYmF0Y2g6IG51bWJlciwgbG9ncz86IExvZ3MpID0+IHZvaWQgfCBQcm9taXNlPHZvaWQ+O1xuICBvbkJhdGNoRW5kPzogKGJhdGNoOiBudW1iZXIsIGxvZ3M/OiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgb25ZaWVsZD86IChlcG9jaDogbnVtYmVyLCBiYXRjaDogbnVtYmVyLCBsb2dzOiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgLy8gVXNlZCBmb3IgdGVzdCBESSBtb2NraW5nLlxuICBub3dGdW5jPzogRnVuY3Rpb247XG4gIG5leHRGcmFtZUZ1bmM/OiBGdW5jdGlvbjtcbn1cblxuLyoqXG4gKiBDdXN0b20gY2FsbGJhY2sgZm9yIHRyYWluaW5nLlxuICovXG5leHBvcnQgY2xhc3MgQ3VzdG9tQ2FsbGJhY2sgZXh0ZW5kcyBCYXNlQ2FsbGJhY2sge1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgdHJhaW5CZWdpbjogKGxvZ3M/OiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHRyYWluRW5kOiAobG9ncz86IExvZ3MpID0+IHZvaWQgfCBQcm9taXNlPHZvaWQ+O1xuICBwcm90ZWN0ZWQgcmVhZG9ubHkgZXBvY2hCZWdpbjpcbiAgICAgIChlcG9jaDogbnVtYmVyLCBsb2dzPzogTG9ncykgPT4gdm9pZCB8IFByb21pc2U8dm9pZD47XG4gIHByb3RlY3RlZCByZWFkb25seSBlcG9jaEVuZDpcbiAgICAgIChlcG9jaDogbnVtYmVyLCBsb2dzPzogTG9ncykgPT4gdm9pZCB8IFByb21pc2U8dm9pZD47XG4gIHByb3RlY3RlZCByZWFkb25seSBiYXRjaEJlZ2luOlxuICAgICAgKGJhdGNoOiBudW1iZXIsIGxvZ3M/OiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IGJhdGNoRW5kOlxuICAgICAgKGJhdGNoOiBudW1iZXIsIGxvZ3M/OiBMb2dzKSA9PiB2b2lkIHwgUHJvbWlzZTx2b2lkPjtcbiAgcHJvdGVjdGVkIHJlYWRvbmx5IHlpZWxkOlxuICAgICAgKGVwb2NoOiBudW1iZXIsIGJhdGNoOiBudW1iZXIsIGxvZ3M6IExvZ3MpID0+IHZvaWQgfCBQcm9taXNlPHZvaWQ+O1xuXG4gIHByaXZhdGUgeWllbGRFdmVyeTogWWllbGRFdmVyeU9wdGlvbnM7XG4gIHByaXZhdGUgY3VycmVudEVwb2NoID0gMDtcbiAgcHVibGljIG5vd0Z1bmM6IEZ1bmN0aW9uO1xuICBwdWJsaWMgbmV4dEZyYW1lRnVuYzogRnVuY3Rpb247XG5cbiAgY29uc3RydWN0b3IoYXJnczogQ3VzdG9tQ2FsbGJhY2tBcmdzLCB5aWVsZEV2ZXJ5PzogWWllbGRFdmVyeU9wdGlvbnMpIHtcbiAgICBzdXBlcigpO1xuICAgIHRoaXMubm93RnVuYyA9IGFyZ3Mubm93RnVuYztcbiAgICB0aGlzLm5leHRGcmFtZUZ1bmMgPSBhcmdzLm5leHRGcmFtZUZ1bmMgfHwgbmV4dEZyYW1lO1xuICAgIHRoaXMueWllbGRFdmVyeSA9IHlpZWxkRXZlcnkgfHwgJ2F1dG8nO1xuICAgIGlmICh0aGlzLnlpZWxkRXZlcnkgPT09ICdhdXRvJykge1xuICAgICAgdGhpcy55aWVsZEV2ZXJ5ID0gREVGQVVMVF9ZSUVMRF9FVkVSWV9NUztcbiAgICB9XG4gICAgaWYgKHRoaXMueWllbGRFdmVyeSA9PT0gJ25ldmVyJyAmJiBhcmdzLm9uWWllbGQgIT0gbnVsbCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICd5aWVsZEV2ZXJ5IGlzIGBuZXZlcmAgYnV0IHlvdSBwcm92aWRlZCBhbiBgb25ZaWVsZGAgY2FsbGJhY2suICcgK1xuICAgICAgICAgICdFaXRoZXIgY2hhbmdlIGB5aWVsZEV2ZXJ5YCBvciByZW1vdmUgdGhlIGNhbGxiYWNrJyk7XG4gICAgfVxuICAgIGlmICh1dGlsLmlzTnVtYmVyKHRoaXMueWllbGRFdmVyeSkpIHtcbiAgICAgIC8vIERlY29yYXRlIGBtYXliZVdhaXRgIHNvIGl0IHdpbGwgYmUgY2FsbGVkIGF0IG1vc3Qgb25jZSBldmVyeVxuICAgICAgLy8gYHlpZWxkRXZlcnlgIG1zLlxuICAgICAgdGhpcy5tYXliZVdhaXQgPSBnZW5lcmljX3V0aWxzLmRlYm91bmNlKFxuICAgICAgICAgIHRoaXMubWF5YmVXYWl0LmJpbmQodGhpcyksIHRoaXMueWllbGRFdmVyeSBhcyBudW1iZXIsIHRoaXMubm93RnVuYyk7XG4gICAgfVxuICAgIHRoaXMudHJhaW5CZWdpbiA9IGFyZ3Mub25UcmFpbkJlZ2luO1xuICAgIHRoaXMudHJhaW5FbmQgPSBhcmdzLm9uVHJhaW5FbmQ7XG4gICAgdGhpcy5lcG9jaEJlZ2luID0gYXJncy5vbkVwb2NoQmVnaW47XG4gICAgdGhpcy5lcG9jaEVuZCA9IGFyZ3Mub25FcG9jaEVuZDtcbiAgICB0aGlzLmJhdGNoQmVnaW4gPSBhcmdzLm9uQmF0Y2hCZWdpbjtcbiAgICB0aGlzLmJhdGNoRW5kID0gYXJncy5vbkJhdGNoRW5kO1xuICAgIHRoaXMueWllbGQgPSBhcmdzLm9uWWllbGQ7XG4gIH1cblxuICBhc3luYyBtYXliZVdhaXQoZXBvY2g6IG51bWJlciwgYmF0Y2g6IG51bWJlciwgbG9nczogVW5yZXNvbHZlZExvZ3MpIHtcbiAgICBjb25zdCBwczogQXJyYXk8dm9pZHxQcm9taXNlPHZvaWQ+PiA9IFtdO1xuICAgIGlmICh0aGlzLnlpZWxkICE9IG51bGwpIHtcbiAgICAgIGF3YWl0IHJlc29sdmVTY2FsYXJzSW5Mb2dzKGxvZ3MpO1xuICAgICAgcHMucHVzaCh0aGlzLnlpZWxkKGVwb2NoLCBiYXRjaCwgbG9ncyBhcyBMb2dzKSk7XG4gICAgfVxuICAgIHBzLnB1c2godGhpcy5uZXh0RnJhbWVGdW5jKCkpO1xuICAgIGF3YWl0IFByb21pc2UuYWxsKHBzKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIG9uRXBvY2hCZWdpbihlcG9jaDogbnVtYmVyLCBsb2dzPzogVW5yZXNvbHZlZExvZ3MpOlxuICAgICAgUHJvbWlzZTx2b2lkPiB7XG4gICAgdGhpcy5jdXJyZW50RXBvY2ggPSBlcG9jaDtcbiAgICBpZiAodGhpcy5lcG9jaEJlZ2luICE9IG51bGwpIHtcbiAgICAgIGF3YWl0IHJlc29sdmVTY2FsYXJzSW5Mb2dzKGxvZ3MpO1xuICAgICAgYXdhaXQgdGhpcy5lcG9jaEJlZ2luKGVwb2NoLCBsb2dzIGFzIExvZ3MpO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIG9uRXBvY2hFbmQoZXBvY2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKTpcbiAgICAgIFByb21pc2U8dm9pZD4ge1xuICAgIGNvbnN0IHBzOiBBcnJheTx2b2lkfFByb21pc2U8dm9pZD4+ID0gW107XG4gICAgaWYgKHRoaXMuZXBvY2hFbmQgIT0gbnVsbCkge1xuICAgICAgYXdhaXQgcmVzb2x2ZVNjYWxhcnNJbkxvZ3MobG9ncyk7XG4gICAgICBwcy5wdXNoKHRoaXMuZXBvY2hFbmQoZXBvY2gsIGxvZ3MgYXMgTG9ncykpO1xuICAgIH1cbiAgICBpZiAodGhpcy55aWVsZEV2ZXJ5ID09PSAnZXBvY2gnKSB7XG4gICAgICBwcy5wdXNoKHRoaXMubmV4dEZyYW1lRnVuYygpKTtcbiAgICB9XG4gICAgYXdhaXQgUHJvbWlzZS5hbGwocHMpO1xuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgb25CYXRjaEJlZ2luKGJhdGNoOiBudW1iZXIsIGxvZ3M/OiBVbnJlc29sdmVkTG9ncyk6XG4gICAgICBQcm9taXNlPHZvaWQ+IHtcbiAgICBpZiAodGhpcy5iYXRjaEJlZ2luICE9IG51bGwpIHtcbiAgICAgIGF3YWl0IHJlc29sdmVTY2FsYXJzSW5Mb2dzKGxvZ3MpO1xuICAgICAgYXdhaXQgdGhpcy5iYXRjaEJlZ2luKGJhdGNoLCBsb2dzIGFzIExvZ3MpO1xuICAgIH1cbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIG9uQmF0Y2hFbmQoYmF0Y2g6IG51bWJlciwgbG9ncz86IFVucmVzb2x2ZWRMb2dzKTpcbiAgICAgIFByb21pc2U8dm9pZD4ge1xuICAgIGNvbnN0IHBzOiBBcnJheTx2b2lkfFByb21pc2U8dm9pZD4+ID0gW107XG4gICAgaWYgKHRoaXMuYmF0Y2hFbmQgIT0gbnVsbCkge1xuICAgICAgYXdhaXQgcmVzb2x2ZVNjYWxhcnNJbkxvZ3MobG9ncyk7XG4gICAgICBwcy5wdXNoKHRoaXMuYmF0Y2hFbmQoYmF0Y2gsIGxvZ3MgYXMgTG9ncykpO1xuICAgIH1cbiAgICBpZiAodGhpcy55aWVsZEV2ZXJ5ID09PSAnYmF0Y2gnKSB7XG4gICAgICBwcy5wdXNoKHRoaXMubmV4dEZyYW1lRnVuYygpKTtcbiAgICB9IGVsc2UgaWYgKHV0aWwuaXNOdW1iZXIodGhpcy55aWVsZEV2ZXJ5KSkge1xuICAgICAgcHMucHVzaCh0aGlzLm1heWJlV2FpdCh0aGlzLmN1cnJlbnRFcG9jaCwgYmF0Y2gsIGxvZ3MpKTtcbiAgICB9XG4gICAgYXdhaXQgUHJvbWlzZS5hbGwocHMpO1xuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgb25UcmFpbkJlZ2luKGxvZ3M/OiBVbnJlc29sdmVkTG9ncyk6IFByb21pc2U8dm9pZD4ge1xuICAgIGlmICh0aGlzLnRyYWluQmVnaW4gIT0gbnVsbCkge1xuICAgICAgYXdhaXQgcmVzb2x2ZVNjYWxhcnNJbkxvZ3MobG9ncyk7XG4gICAgICBhd2FpdCB0aGlzLnRyYWluQmVnaW4obG9ncyBhcyBMb2dzKTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyBvblRyYWluRW5kKGxvZ3M/OiBVbnJlc29sdmVkTG9ncyk6IFByb21pc2U8dm9pZD4ge1xuICAgIGlmICh0aGlzLnRyYWluRW5kICE9IG51bGwpIHtcbiAgICAgIGF3YWl0IHJlc29sdmVTY2FsYXJzSW5Mb2dzKGxvZ3MpO1xuICAgICAgYXdhaXQgdGhpcy50cmFpbkVuZChsb2dzIGFzIExvZ3MpO1xuICAgIH1cbiAgfVxufVxuXG4vKipcbiAqIFN0YW5kYXJkaXplIGNhbGxiYWNrcyBvciBjb25maWd1cmF0aW9ucyBvZiB0aGVtIHRvIGFuIEFycmF5IG9mIGNhbGxiYWNrcy5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHN0YW5kYXJkaXplQ2FsbGJhY2tzKFxuICAgIGNhbGxiYWNrczogQmFzZUNhbGxiYWNrfEJhc2VDYWxsYmFja1tdfEN1c3RvbUNhbGxiYWNrQXJnc3xcbiAgICBDdXN0b21DYWxsYmFja0FyZ3NbXSxcbiAgICB5aWVsZEV2ZXJ5OiBZaWVsZEV2ZXJ5T3B0aW9ucyk6IEJhc2VDYWxsYmFja1tdIHtcbiAgaWYgKGNhbGxiYWNrcyA9PSBudWxsKSB7XG4gICAgY2FsbGJhY2tzID0ge30gYXMgQmFzZUNhbGxiYWNrO1xuICB9XG4gIGlmIChjYWxsYmFja3MgaW5zdGFuY2VvZiBCYXNlQ2FsbGJhY2spIHtcbiAgICByZXR1cm4gW2NhbGxiYWNrc107XG4gIH1cbiAgaWYgKEFycmF5LmlzQXJyYXkoY2FsbGJhY2tzKSAmJiBjYWxsYmFja3NbMF0gaW5zdGFuY2VvZiBCYXNlQ2FsbGJhY2spIHtcbiAgICByZXR1cm4gY2FsbGJhY2tzIGFzIEJhc2VDYWxsYmFja1tdO1xuICB9XG4gIC8vIENvbnZlcnQgY3VzdG9tIGNhbGxiYWNrIGNvbmZpZ3MgdG8gY3VzdG9tIGNhbGxiYWNrIG9iamVjdHMuXG4gIGNvbnN0IGNhbGxiYWNrQ29uZmlncyA9XG4gICAgICBnZW5lcmljX3V0aWxzLnRvTGlzdChjYWxsYmFja3MpIGFzIEN1c3RvbUNhbGxiYWNrQXJnc1tdO1xuICByZXR1cm4gY2FsbGJhY2tDb25maWdzLm1hcChcbiAgICAgIGNhbGxiYWNrQ29uZmlnID0+IG5ldyBDdXN0b21DYWxsYmFjayhjYWxsYmFja0NvbmZpZywgeWllbGRFdmVyeSkpO1xufVxuXG5leHBvcnQgZGVjbGFyZSB0eXBlIEJhc2VDYWxsYmFja0NvbnN0cnVjdG9yID0ge1xuICBuZXcgKCk6IEJhc2VDYWxsYmFja1xufTtcblxuLyoqXG4gKiBBIGdsb2JhbCByZWdpc3RyeSBmb3IgY2FsbGJhY2sgY29uc3RydWN0b3JzIHRvIGJlIHVzZWQgZHVyaW5nXG4gKiBMYXllcnNNb2RlbC5maXQoKS5cbiAqL1xuZXhwb3J0IGNsYXNzIENhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeSB7XG4gIHByaXZhdGUgc3RhdGljIGNvbnN0cnVjdG9yczpcbiAgICAgIHtbdmVyYm9zaXR5TGV2ZWw6IG51bWJlcl06IEJhc2VDYWxsYmFja0NvbnN0cnVjdG9yW119ID0ge307XG5cbiAgLyoqXG4gICAqIEJsb2NrcyBwdWJsaWMgYWNjZXNzIHRvIGNvbnN0cnVjdG9yLlxuICAgKi9cbiAgcHJpdmF0ZSBjb25zdHJ1Y3RvcigpIHt9XG5cbiAgLyoqXG4gICAqIFJlZ2lzdGVyIGEgdGYuTGF5ZXJzTW9kZWwuZml0KCkgY2FsbGJhY2sgY29uc3RydWN0b3IuXG4gICAqXG4gICAqIFRoZSByZWdpc3RlcmVkIGNhbGxiYWNrIGNvbnN0cnVjdG9yIHdpbGwgYmUgdXNlZCB0byBpbnN0YW50aWF0ZVxuICAgKiBjYWxsYmFja3MgZm9yIGV2ZXJ5IHRmLkxheWVyc01vZGVsLmZpdCgpIGNhbGwgYWZ0ZXJ3YXJkcy5cbiAgICpcbiAgICogQHBhcmFtIHZlcmJvc2l0eUxldmVsIExldmVsIG9mIHZlcmJvc2l0eSBhdCB3aGljaCB0aGUgYGNhbGxiYWNrQ29uc3RydWN0b3JgXG4gICAqICAgaXMgdG8gYmUgcmVpZ3N0ZXJlZC5cbiAgICogQHBhcmFtIGNhbGxiYWNrQ29uc3RydWN0b3IgQSBuby1hcmcgY29uc3RydWN0b3IgZm9yIGB0Zi5DYWxsYmFja2AuXG4gICAqIEB0aHJvd3MgRXJyb3IsIGlmIHRoZSBzYW1lIGNhbGxiYWNrQ29uc3RydWN0b3IgaGFzIGJlZW4gcmVnaXN0ZXJlZCBiZWZvcmUsXG4gICAqICAgZWl0aGVyIGF0IHRoZSBzYW1lIG9yIGEgZGlmZmVyZW50IGB2ZXJib3NpdHlMZXZlbGAuXG4gICAqL1xuICBzdGF0aWMgcmVnaXN0ZXJDYWxsYmFja0NvbnN0cnVjdG9yKFxuICAgICAgdmVyYm9zaXR5TGV2ZWw6IG51bWJlciwgY2FsbGJhY2tDb25zdHJ1Y3RvcjogQmFzZUNhbGxiYWNrQ29uc3RydWN0b3IpIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdmVyYm9zaXR5TGV2ZWwgPj0gMCAmJiBOdW1iZXIuaXNJbnRlZ2VyKHZlcmJvc2l0eUxldmVsKSxcbiAgICAgICAgKCkgPT4gYFZlcmJvc2l0eSBsZXZlbCBpcyBleHBlY3RlZCB0byBiZSBhbiBpbnRlZ2VyID49IDAsIGAgK1xuICAgICAgICAgICAgYGJ1dCBnb3QgJHt2ZXJib3NpdHlMZXZlbH1gKTtcbiAgICBDYWxsYmFja0NvbnN0cnVjdG9yUmVnaXN0cnkuY2hlY2tGb3JEdXBsaWNhdGUoY2FsbGJhY2tDb25zdHJ1Y3Rvcik7XG4gICAgaWYgKENhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jb25zdHJ1Y3RvcnNbdmVyYm9zaXR5TGV2ZWxdID09IG51bGwpIHtcbiAgICAgIENhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jb25zdHJ1Y3RvcnNbdmVyYm9zaXR5TGV2ZWxdID0gW107XG4gICAgfVxuICAgIENhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jb25zdHJ1Y3RvcnNbdmVyYm9zaXR5TGV2ZWxdLnB1c2goXG4gICAgICAgIGNhbGxiYWNrQ29uc3RydWN0b3IpO1xuICB9XG5cbiAgcHJpdmF0ZSBzdGF0aWMgY2hlY2tGb3JEdXBsaWNhdGUoY2FsbGJhY2tDb25zdHJ1Y3RvcjpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEJhc2VDYWxsYmFja0NvbnN0cnVjdG9yKSB7XG4gICAgZm9yIChjb25zdCBsZXZlbE5hbWUgaW4gQ2FsbGJhY2tDb25zdHJ1Y3RvclJlZ2lzdHJ5LmNvbnN0cnVjdG9ycykge1xuICAgICAgY29uc3QgY29uc3RydWN0b3JzID0gQ2FsbGJhY2tDb25zdHJ1Y3RvclJlZ2lzdHJ5LmNvbnN0cnVjdG9yc1srbGV2ZWxOYW1lXTtcbiAgICAgIGNvbnN0cnVjdG9ycy5mb3JFYWNoKGN0b3IgPT4ge1xuICAgICAgICBpZiAoY3RvciA9PT0gY2FsbGJhY2tDb25zdHJ1Y3Rvcikge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKCdEdXBsaWNhdGUgY2FsbGJhY2sgY29uc3RydWN0b3IuJyk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cbiAgfVxuXG4gIC8qKlxuICAgKiBDbGVhciBhbGwgcmVnaXN0ZXJlZCBjYWxsYmFjayBjb25zdHJ1Y3RvcnMuXG4gICAqL1xuICBwcm90ZWN0ZWQgc3RhdGljIGNsZWFyKCkge1xuICAgIENhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jb25zdHJ1Y3RvcnMgPSB7fTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDcmVhdGUgY2FsbGJhY2tzIHVzaW5nIHRoZSByZWdpc3RlcmVkIGNhbGxiYWNrIGNvbnN0cnVjdG9ycy5cbiAgICpcbiAgICogR2l2ZW4gYHZlcmJvc2l0eUxldmVsYCwgYWxsIGNvbnN0cnVjdG9ycyByZWdpc3RlcmVkIGF0IHRoYXQgbGV2ZWwgb3IgYWJvdmVcbiAgICogd2lsbCBiZSBjYWxsZWQgYW5kIHRoZSBpbnN0YW50aWF0ZWQgY2FsbGJhY2tzIHdpbGwgYmUgdXNlZC5cbiAgICpcbiAgICogQHBhcmFtIHZlcmJvc2l0eUxldmVsOiBMZXZlbCBvZiB2ZXJib3NpdHkuXG4gICAqL1xuICBzdGF0aWMgY3JlYXRlQ2FsbGJhY2tzKHZlcmJvc2l0eUxldmVsOiBudW1iZXIpOiBCYXNlQ2FsbGJhY2tbXSB7XG4gICAgY29uc3QgY29uc3RydWN0b3JzOiBCYXNlQ2FsbGJhY2tDb25zdHJ1Y3RvcltdID0gW107XG4gICAgZm9yIChjb25zdCBsZXZlbE5hbWUgaW4gQ2FsbGJhY2tDb25zdHJ1Y3RvclJlZ2lzdHJ5LmNvbnN0cnVjdG9ycykge1xuICAgICAgY29uc3QgbGV2ZWwgPSArbGV2ZWxOYW1lO1xuICAgICAgaWYgKHZlcmJvc2l0eUxldmVsID49IGxldmVsKSB7XG4gICAgICAgIGNvbnN0cnVjdG9ycy5wdXNoKC4uLkNhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jb25zdHJ1Y3RvcnNbbGV2ZWxdKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGNvbnN0cnVjdG9ycy5tYXAoY3RvciA9PiBuZXcgY3RvcigpKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY29uZmlndXJlQ2FsbGJhY2tzKFxuICAgIGNhbGxiYWNrczogQmFzZUNhbGxiYWNrW10sIHZlcmJvc2U6IE1vZGVsTG9nZ2luZ1ZlcmJvc2l0eSwgZXBvY2hzOiBudW1iZXIsXG4gICAgaW5pdGlhbEVwb2NoOiBudW1iZXIsIG51bVRyYWluU2FtcGxlczogbnVtYmVyLCBzdGVwc1BlckVwb2NoOiBudW1iZXIsXG4gICAgYmF0Y2hTaXplOiBudW1iZXIsIGRvVmFsaWRhdGlvbjogYm9vbGVhbixcbiAgICBjYWxsYmFja01ldHJpY3M6IHN0cmluZ1tdKToge2NhbGxiYWNrTGlzdDogQ2FsbGJhY2tMaXN0LCBoaXN0b3J5OiBIaXN0b3J5fSB7XG4gIGNvbnN0IGhpc3RvcnkgPSBuZXcgSGlzdG9yeSgpO1xuICBjb25zdCBhY3R1YWxDYWxsYmFja3M6IEJhc2VDYWxsYmFja1tdID0gW1xuICAgIG5ldyBCYXNlTG9nZ2VyKCksIC4uLkNhbGxiYWNrQ29uc3RydWN0b3JSZWdpc3RyeS5jcmVhdGVDYWxsYmFja3ModmVyYm9zZSlcbiAgXTtcbiAgaWYgKGNhbGxiYWNrcyAhPSBudWxsKSB7XG4gICAgYWN0dWFsQ2FsbGJhY2tzLnB1c2goLi4uY2FsbGJhY2tzKTtcbiAgfVxuICBhY3R1YWxDYWxsYmFja3MucHVzaChoaXN0b3J5KTtcbiAgY29uc3QgY2FsbGJhY2tMaXN0ID0gbmV3IENhbGxiYWNrTGlzdChhY3R1YWxDYWxsYmFja3MpO1xuXG4gIC8vIFRPRE8oY2Fpcyk6IEZpZ3VyZSBvdXQgd2hlbiB0aGlzIExheWVyc01vZGVsIGluc3RhbmNlIGNhbiBoYXZlIGFcbiAgLy8gZHluYW1pY2FsbHlcbiAgLy8gICBzZXQgcHJvcGVydHkgY2FsbGVkICdjYWxsYmFja19tb2RlbCcgYXMgaW4gUHlLZXJhcy5cblxuICBjYWxsYmFja0xpc3Quc2V0UGFyYW1zKHtcbiAgICBlcG9jaHMsXG4gICAgaW5pdGlhbEVwb2NoLFxuICAgIHNhbXBsZXM6IG51bVRyYWluU2FtcGxlcyxcbiAgICBzdGVwczogc3RlcHNQZXJFcG9jaCxcbiAgICBiYXRjaFNpemUsXG4gICAgdmVyYm9zZSxcbiAgICBkb1ZhbGlkYXRpb24sXG4gICAgbWV0cmljczogY2FsbGJhY2tNZXRyaWNzLFxuICB9KTtcbiAgcmV0dXJuIHtjYWxsYmFja0xpc3QsIGhpc3Rvcnl9O1xufVxuIl19