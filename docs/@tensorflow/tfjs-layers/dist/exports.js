/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/**
 * Exported functions.
 */
import { CallbackConstructorRegistry } from './base_callbacks';
import { Input, } from './engine/input_layer';
import { LayersModel } from './engine/training';
import { Sequential } from './models';
export { loadLayersModel } from './models';
// TODO(cais): Add doc string to all the public static functions in this
//   class; include exectuable JavaScript code snippets where applicable
//   (b/74074458).
// LayersModel and related factory methods.
/**
 * A model is a data structure that consists of `Layers` and defines inputs
 * and outputs.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.model` is more generic, supporting an arbitrary graph (without
 * cycles) of layers. `tf.sequential` is less generic and supports only a linear
 * stack of layers.
 *
 * When creating a `tf.LayersModel`, specify its input(s) and output(s). Layers
 * are used to wire input(s) to output(s).
 *
 * For example, the following code snippet defines a model consisting of
 * two `dense` layers, with 10 and 4 units, respectively.
 *
 * ```js
 * // Define input, which has a size of 5 (not including batch dimension).
 * const input = tf.input({shape: [5]});
 *
 * // First dense layer uses relu activation.
 * const denseLayer1 = tf.layers.dense({units: 10, activation: 'relu'});
 * // Second dense layer uses softmax activation.
 * const denseLayer2 = tf.layers.dense({units: 4, activation: 'softmax'});
 *
 * // Obtain the output symbolic tensor by applying the layers on the input.
 * const output = denseLayer2.apply(denseLayer1.apply(input));
 *
 * // Create the model based on the inputs.
 * const model = tf.model({inputs: input, outputs: output});
 *
 * // The model can be used for training, evaluation and prediction.
 * // For example, the following line runs prediction with the model on
 * // some fake data.
 * model.predict(tf.ones([2, 5])).print();
 * ```
 * See also:
 *   `tf.sequential`, `tf.loadLayersModel`.
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
export function model(args) {
    return new LayersModel(args);
}
/**
 * Creates a `tf.Sequential` model.  A sequential model is any model where the
 * outputs of one layer are the inputs to the next layer, i.e. the model
 * topology is a simple 'stack' of layers, with no branching or skipping.
 *
 * This means that the first layer passed to a `tf.Sequential` model should have
 * a defined input shape. What that means is that it should have received an
 * `inputShape` or `batchInputShape` argument, or for some type of layers
 * (recurrent, Dense...) an `inputDim` argument.
 *
 * The key difference between `tf.model` and `tf.sequential` is that
 * `tf.sequential` is less generic, supporting only a linear stack of layers.
 * `tf.model` is more generic and supports an arbitrary graph (without
 * cycles) of layers.
 *
 * Examples:
 *
 * ```js
 * const model = tf.sequential();
 *
 * // First layer must have an input shape defined.
 * model.add(tf.layers.dense({units: 32, inputShape: [50]}));
 * // Afterwards, TF.js does automatic shape inference.
 * model.add(tf.layers.dense({units: 4}));
 *
 * // Inspect the inferred shape of the model's output, which equals
 * // `[null, 4]`. The 1st dimension is the undetermined batch dimension; the
 * // 2nd is the output size of the model's last layer.
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * It is also possible to specify a batch size (with potentially undetermined
 * batch dimension, denoted by "null") for the first layer using the
 * `batchInputShape` key. The following example is equivalent to the above:
 *
 * ```js
 * const model = tf.sequential();
 *
 * // First layer must have a defined input shape
 * model.add(tf.layers.dense({units: 32, batchInputShape: [null, 50]}));
 * // Afterwards, TF.js does automatic shape inference.
 * model.add(tf.layers.dense({units: 4}));
 *
 * // Inspect the inferred shape of the model's output.
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * You can also use an `Array` of already-constructed `Layer`s to create
 * a `tf.Sequential` model:
 *
 * ```js
 * const model = tf.sequential({
 *   layers: [tf.layers.dense({units: 32, inputShape: [50]}),
 *            tf.layers.dense({units: 4})]
 * });
 * console.log(JSON.stringify(model.outputs[0].shape));
 * ```
 *
 * @doc {heading: 'Models', subheading: 'Creation'}
 */
export function sequential(config) {
    return new Sequential(config);
}
/**
 * Used to instantiate an input to a model as a `tf.SymbolicTensor`.
 *
 * Users should call the `input` factory function for
 * consistency with other generator functions.
 *
 * Example:
 *
 * ```js
 * // Defines a simple logistic regression model with 32 dimensional input
 * // and 3 dimensional output.
 * const x = tf.input({shape: [32]});
 * const y = tf.layers.dense({units: 3, activation: 'softmax'}).apply(x);
 * const model = tf.model({inputs: x, outputs: y});
 * model.predict(tf.ones([2, 32])).print();
 * ```
 *
 * Note: `input` is only necessary when using `model`. When using
 * `sequential`, specify `inputShape` for the first layer or use `inputLayer`
 * as the first layer.
 *
 * @doc {heading: 'Models', subheading: 'Inputs'}
 */
export function input(config) {
    return Input(config);
}
export function registerCallbackConstructor(verbosityLevel, callbackConstructor) {
    CallbackConstructorRegistry.registerCallbackConstructor(verbosityLevel, callbackConstructor);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhwb3J0cy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9leHBvcnRzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEVBQTBCLDJCQUEyQixFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFFdEYsT0FBTyxFQUFDLEtBQUssR0FBZSxNQUFNLHNCQUFzQixDQUFDO0FBRXpELE9BQU8sRUFBQyxXQUFXLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUM5QyxPQUFPLEVBQUMsVUFBVSxFQUFpQixNQUFNLFVBQVUsQ0FBQztBQUVwRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBRXpDLHdFQUF3RTtBQUN4RSx3RUFBd0U7QUFDeEUsa0JBQWtCO0FBRWxCLDJDQUEyQztBQUUzQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBdUNHO0FBQ0gsTUFBTSxVQUFVLEtBQUssQ0FBQyxJQUFtQjtJQUN2QyxPQUFPLElBQUksV0FBVyxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQy9CLENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0EyREc7QUFDSCxNQUFNLFVBQVUsVUFBVSxDQUFDLE1BQXVCO0lBQ2hELE9BQU8sSUFBSSxVQUFVLENBQUMsTUFBTSxDQUFDLENBQUM7QUFDaEMsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBc0JHO0FBQ0gsTUFBTSxVQUFVLEtBQUssQ0FBQyxNQUFtQjtJQUN2QyxPQUFPLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztBQUN2QixDQUFDO0FBRUQsTUFBTSxVQUFVLDJCQUEyQixDQUN2QyxjQUFzQixFQUN0QixtQkFBNEM7SUFDOUMsMkJBQTJCLENBQUMsMkJBQTJCLENBQ25ELGNBQWMsRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO0FBQzNDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKipcbiAqIEV4cG9ydGVkIGZ1bmN0aW9ucy5cbiAqL1xuXG5pbXBvcnQge0Jhc2VDYWxsYmFja0NvbnN0cnVjdG9yLCBDYWxsYmFja0NvbnN0cnVjdG9yUmVnaXN0cnl9IGZyb20gJy4vYmFzZV9jYWxsYmFja3MnO1xuaW1wb3J0IHtDb250YWluZXJBcmdzfSBmcm9tICcuL2VuZ2luZS9jb250YWluZXInO1xuaW1wb3J0IHtJbnB1dCwgSW5wdXRDb25maWcsfSBmcm9tICcuL2VuZ2luZS9pbnB1dF9sYXllcic7XG5pbXBvcnQge1N5bWJvbGljVGVuc29yfSBmcm9tICcuL2VuZ2luZS90b3BvbG9neSc7XG5pbXBvcnQge0xheWVyc01vZGVsfSBmcm9tICcuL2VuZ2luZS90cmFpbmluZyc7XG5pbXBvcnQge1NlcXVlbnRpYWwsIFNlcXVlbnRpYWxBcmdzfSBmcm9tICcuL21vZGVscyc7XG5cbmV4cG9ydCB7bG9hZExheWVyc01vZGVsfSBmcm9tICcuL21vZGVscyc7XG5cbi8vIFRPRE8oY2Fpcyk6IEFkZCBkb2Mgc3RyaW5nIHRvIGFsbCB0aGUgcHVibGljIHN0YXRpYyBmdW5jdGlvbnMgaW4gdGhpc1xuLy8gICBjbGFzczsgaW5jbHVkZSBleGVjdHVhYmxlIEphdmFTY3JpcHQgY29kZSBzbmlwcGV0cyB3aGVyZSBhcHBsaWNhYmxlXG4vLyAgIChiLzc0MDc0NDU4KS5cblxuLy8gTGF5ZXJzTW9kZWwgYW5kIHJlbGF0ZWQgZmFjdG9yeSBtZXRob2RzLlxuXG4vKipcbiAqIEEgbW9kZWwgaXMgYSBkYXRhIHN0cnVjdHVyZSB0aGF0IGNvbnNpc3RzIG9mIGBMYXllcnNgIGFuZCBkZWZpbmVzIGlucHV0c1xuICogYW5kIG91dHB1dHMuXG4gKlxuICogVGhlIGtleSBkaWZmZXJlbmNlIGJldHdlZW4gYHRmLm1vZGVsYCBhbmQgYHRmLnNlcXVlbnRpYWxgIGlzIHRoYXRcbiAqIGB0Zi5tb2RlbGAgaXMgbW9yZSBnZW5lcmljLCBzdXBwb3J0aW5nIGFuIGFyYml0cmFyeSBncmFwaCAod2l0aG91dFxuICogY3ljbGVzKSBvZiBsYXllcnMuIGB0Zi5zZXF1ZW50aWFsYCBpcyBsZXNzIGdlbmVyaWMgYW5kIHN1cHBvcnRzIG9ubHkgYSBsaW5lYXJcbiAqIHN0YWNrIG9mIGxheWVycy5cbiAqXG4gKiBXaGVuIGNyZWF0aW5nIGEgYHRmLkxheWVyc01vZGVsYCwgc3BlY2lmeSBpdHMgaW5wdXQocykgYW5kIG91dHB1dChzKS4gTGF5ZXJzXG4gKiBhcmUgdXNlZCB0byB3aXJlIGlucHV0KHMpIHRvIG91dHB1dChzKS5cbiAqXG4gKiBGb3IgZXhhbXBsZSwgdGhlIGZvbGxvd2luZyBjb2RlIHNuaXBwZXQgZGVmaW5lcyBhIG1vZGVsIGNvbnNpc3Rpbmcgb2ZcbiAqIHR3byBgZGVuc2VgIGxheWVycywgd2l0aCAxMCBhbmQgNCB1bml0cywgcmVzcGVjdGl2ZWx5LlxuICpcbiAqIGBgYGpzXG4gKiAvLyBEZWZpbmUgaW5wdXQsIHdoaWNoIGhhcyBhIHNpemUgb2YgNSAobm90IGluY2x1ZGluZyBiYXRjaCBkaW1lbnNpb24pLlxuICogY29uc3QgaW5wdXQgPSB0Zi5pbnB1dCh7c2hhcGU6IFs1XX0pO1xuICpcbiAqIC8vIEZpcnN0IGRlbnNlIGxheWVyIHVzZXMgcmVsdSBhY3RpdmF0aW9uLlxuICogY29uc3QgZGVuc2VMYXllcjEgPSB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAxMCwgYWN0aXZhdGlvbjogJ3JlbHUnfSk7XG4gKiAvLyBTZWNvbmQgZGVuc2UgbGF5ZXIgdXNlcyBzb2Z0bWF4IGFjdGl2YXRpb24uXG4gKiBjb25zdCBkZW5zZUxheWVyMiA9IHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDQsIGFjdGl2YXRpb246ICdzb2Z0bWF4J30pO1xuICpcbiAqIC8vIE9idGFpbiB0aGUgb3V0cHV0IHN5bWJvbGljIHRlbnNvciBieSBhcHBseWluZyB0aGUgbGF5ZXJzIG9uIHRoZSBpbnB1dC5cbiAqIGNvbnN0IG91dHB1dCA9IGRlbnNlTGF5ZXIyLmFwcGx5KGRlbnNlTGF5ZXIxLmFwcGx5KGlucHV0KSk7XG4gKlxuICogLy8gQ3JlYXRlIHRoZSBtb2RlbCBiYXNlZCBvbiB0aGUgaW5wdXRzLlxuICogY29uc3QgbW9kZWwgPSB0Zi5tb2RlbCh7aW5wdXRzOiBpbnB1dCwgb3V0cHV0czogb3V0cHV0fSk7XG4gKlxuICogLy8gVGhlIG1vZGVsIGNhbiBiZSB1c2VkIGZvciB0cmFpbmluZywgZXZhbHVhdGlvbiBhbmQgcHJlZGljdGlvbi5cbiAqIC8vIEZvciBleGFtcGxlLCB0aGUgZm9sbG93aW5nIGxpbmUgcnVucyBwcmVkaWN0aW9uIHdpdGggdGhlIG1vZGVsIG9uXG4gKiAvLyBzb21lIGZha2UgZGF0YS5cbiAqIG1vZGVsLnByZWRpY3QodGYub25lcyhbMiwgNV0pKS5wcmludCgpO1xuICogYGBgXG4gKiBTZWUgYWxzbzpcbiAqICAgYHRmLnNlcXVlbnRpYWxgLCBgdGYubG9hZExheWVyc01vZGVsYC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTW9kZWxzJywgc3ViaGVhZGluZzogJ0NyZWF0aW9uJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1vZGVsKGFyZ3M6IENvbnRhaW5lckFyZ3MpOiBMYXllcnNNb2RlbCB7XG4gIHJldHVybiBuZXcgTGF5ZXJzTW9kZWwoYXJncyk7XG59XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5TZXF1ZW50aWFsYCBtb2RlbC4gIEEgc2VxdWVudGlhbCBtb2RlbCBpcyBhbnkgbW9kZWwgd2hlcmUgdGhlXG4gKiBvdXRwdXRzIG9mIG9uZSBsYXllciBhcmUgdGhlIGlucHV0cyB0byB0aGUgbmV4dCBsYXllciwgaS5lLiB0aGUgbW9kZWxcbiAqIHRvcG9sb2d5IGlzIGEgc2ltcGxlICdzdGFjaycgb2YgbGF5ZXJzLCB3aXRoIG5vIGJyYW5jaGluZyBvciBza2lwcGluZy5cbiAqXG4gKiBUaGlzIG1lYW5zIHRoYXQgdGhlIGZpcnN0IGxheWVyIHBhc3NlZCB0byBhIGB0Zi5TZXF1ZW50aWFsYCBtb2RlbCBzaG91bGQgaGF2ZVxuICogYSBkZWZpbmVkIGlucHV0IHNoYXBlLiBXaGF0IHRoYXQgbWVhbnMgaXMgdGhhdCBpdCBzaG91bGQgaGF2ZSByZWNlaXZlZCBhblxuICogYGlucHV0U2hhcGVgIG9yIGBiYXRjaElucHV0U2hhcGVgIGFyZ3VtZW50LCBvciBmb3Igc29tZSB0eXBlIG9mIGxheWVyc1xuICogKHJlY3VycmVudCwgRGVuc2UuLi4pIGFuIGBpbnB1dERpbWAgYXJndW1lbnQuXG4gKlxuICogVGhlIGtleSBkaWZmZXJlbmNlIGJldHdlZW4gYHRmLm1vZGVsYCBhbmQgYHRmLnNlcXVlbnRpYWxgIGlzIHRoYXRcbiAqIGB0Zi5zZXF1ZW50aWFsYCBpcyBsZXNzIGdlbmVyaWMsIHN1cHBvcnRpbmcgb25seSBhIGxpbmVhciBzdGFjayBvZiBsYXllcnMuXG4gKiBgdGYubW9kZWxgIGlzIG1vcmUgZ2VuZXJpYyBhbmQgc3VwcG9ydHMgYW4gYXJiaXRyYXJ5IGdyYXBoICh3aXRob3V0XG4gKiBjeWNsZXMpIG9mIGxheWVycy5cbiAqXG4gKiBFeGFtcGxlczpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKlxuICogLy8gRmlyc3QgbGF5ZXIgbXVzdCBoYXZlIGFuIGlucHV0IHNoYXBlIGRlZmluZWQuXG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLmRlbnNlKHt1bml0czogMzIsIGlucHV0U2hhcGU6IFs1MF19KSk7XG4gKiAvLyBBZnRlcndhcmRzLCBURi5qcyBkb2VzIGF1dG9tYXRpYyBzaGFwZSBpbmZlcmVuY2UuXG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLmRlbnNlKHt1bml0czogNH0pKTtcbiAqXG4gKiAvLyBJbnNwZWN0IHRoZSBpbmZlcnJlZCBzaGFwZSBvZiB0aGUgbW9kZWwncyBvdXRwdXQsIHdoaWNoIGVxdWFsc1xuICogLy8gYFtudWxsLCA0XWAuIFRoZSAxc3QgZGltZW5zaW9uIGlzIHRoZSB1bmRldGVybWluZWQgYmF0Y2ggZGltZW5zaW9uOyB0aGVcbiAqIC8vIDJuZCBpcyB0aGUgb3V0cHV0IHNpemUgb2YgdGhlIG1vZGVsJ3MgbGFzdCBsYXllci5cbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG1vZGVsLm91dHB1dHNbMF0uc2hhcGUpKTtcbiAqIGBgYFxuICpcbiAqIEl0IGlzIGFsc28gcG9zc2libGUgdG8gc3BlY2lmeSBhIGJhdGNoIHNpemUgKHdpdGggcG90ZW50aWFsbHkgdW5kZXRlcm1pbmVkXG4gKiBiYXRjaCBkaW1lbnNpb24sIGRlbm90ZWQgYnkgXCJudWxsXCIpIGZvciB0aGUgZmlyc3QgbGF5ZXIgdXNpbmcgdGhlXG4gKiBgYmF0Y2hJbnB1dFNoYXBlYCBrZXkuIFRoZSBmb2xsb3dpbmcgZXhhbXBsZSBpcyBlcXVpdmFsZW50IHRvIHRoZSBhYm92ZTpcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgbW9kZWwgPSB0Zi5zZXF1ZW50aWFsKCk7XG4gKlxuICogLy8gRmlyc3QgbGF5ZXIgbXVzdCBoYXZlIGEgZGVmaW5lZCBpbnB1dCBzaGFwZVxuICogbW9kZWwuYWRkKHRmLmxheWVycy5kZW5zZSh7dW5pdHM6IDMyLCBiYXRjaElucHV0U2hhcGU6IFtudWxsLCA1MF19KSk7XG4gKiAvLyBBZnRlcndhcmRzLCBURi5qcyBkb2VzIGF1dG9tYXRpYyBzaGFwZSBpbmZlcmVuY2UuXG4gKiBtb2RlbC5hZGQodGYubGF5ZXJzLmRlbnNlKHt1bml0czogNH0pKTtcbiAqXG4gKiAvLyBJbnNwZWN0IHRoZSBpbmZlcnJlZCBzaGFwZSBvZiB0aGUgbW9kZWwncyBvdXRwdXQuXG4gKiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShtb2RlbC5vdXRwdXRzWzBdLnNoYXBlKSk7XG4gKiBgYGBcbiAqXG4gKiBZb3UgY2FuIGFsc28gdXNlIGFuIGBBcnJheWAgb2YgYWxyZWFkeS1jb25zdHJ1Y3RlZCBgTGF5ZXJgcyB0byBjcmVhdGVcbiAqIGEgYHRmLlNlcXVlbnRpYWxgIG1vZGVsOlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBtb2RlbCA9IHRmLnNlcXVlbnRpYWwoe1xuICogICBsYXllcnM6IFt0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAzMiwgaW5wdXRTaGFwZTogWzUwXX0pLFxuICogICAgICAgICAgICB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiA0fSldXG4gKiB9KTtcbiAqIGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KG1vZGVsLm91dHB1dHNbMF0uc2hhcGUpKTtcbiAqIGBgYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNb2RlbHMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5leHBvcnQgZnVuY3Rpb24gc2VxdWVudGlhbChjb25maWc/OiBTZXF1ZW50aWFsQXJncyk6IFNlcXVlbnRpYWwge1xuICByZXR1cm4gbmV3IFNlcXVlbnRpYWwoY29uZmlnKTtcbn1cblxuLyoqXG4gKiBVc2VkIHRvIGluc3RhbnRpYXRlIGFuIGlucHV0IHRvIGEgbW9kZWwgYXMgYSBgdGYuU3ltYm9saWNUZW5zb3JgLlxuICpcbiAqIFVzZXJzIHNob3VsZCBjYWxsIHRoZSBgaW5wdXRgIGZhY3RvcnkgZnVuY3Rpb24gZm9yXG4gKiBjb25zaXN0ZW5jeSB3aXRoIG90aGVyIGdlbmVyYXRvciBmdW5jdGlvbnMuXG4gKlxuICogRXhhbXBsZTpcbiAqXG4gKiBgYGBqc1xuICogLy8gRGVmaW5lcyBhIHNpbXBsZSBsb2dpc3RpYyByZWdyZXNzaW9uIG1vZGVsIHdpdGggMzIgZGltZW5zaW9uYWwgaW5wdXRcbiAqIC8vIGFuZCAzIGRpbWVuc2lvbmFsIG91dHB1dC5cbiAqIGNvbnN0IHggPSB0Zi5pbnB1dCh7c2hhcGU6IFszMl19KTtcbiAqIGNvbnN0IHkgPSB0Zi5sYXllcnMuZGVuc2Uoe3VuaXRzOiAzLCBhY3RpdmF0aW9uOiAnc29mdG1heCd9KS5hcHBseSh4KTtcbiAqIGNvbnN0IG1vZGVsID0gdGYubW9kZWwoe2lucHV0czogeCwgb3V0cHV0czogeX0pO1xuICogbW9kZWwucHJlZGljdCh0Zi5vbmVzKFsyLCAzMl0pKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogTm90ZTogYGlucHV0YCBpcyBvbmx5IG5lY2Vzc2FyeSB3aGVuIHVzaW5nIGBtb2RlbGAuIFdoZW4gdXNpbmdcbiAqIGBzZXF1ZW50aWFsYCwgc3BlY2lmeSBgaW5wdXRTaGFwZWAgZm9yIHRoZSBmaXJzdCBsYXllciBvciB1c2UgYGlucHV0TGF5ZXJgXG4gKiBhcyB0aGUgZmlyc3QgbGF5ZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01vZGVscycsIHN1YmhlYWRpbmc6ICdJbnB1dHMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gaW5wdXQoY29uZmlnOiBJbnB1dENvbmZpZyk6IFN5bWJvbGljVGVuc29yIHtcbiAgcmV0dXJuIElucHV0KGNvbmZpZyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiByZWdpc3RlckNhbGxiYWNrQ29uc3RydWN0b3IoXG4gICAgdmVyYm9zaXR5TGV2ZWw6IG51bWJlcixcbiAgICBjYWxsYmFja0NvbnN0cnVjdG9yOiBCYXNlQ2FsbGJhY2tDb25zdHJ1Y3Rvcik6IHZvaWQge1xuICBDYWxsYmFja0NvbnN0cnVjdG9yUmVnaXN0cnkucmVnaXN0ZXJDYWxsYmFja0NvbnN0cnVjdG9yKFxuICAgICAgdmVyYm9zaXR5TGV2ZWwsIGNhbGxiYWNrQ29uc3RydWN0b3IpO1xufVxuIl19