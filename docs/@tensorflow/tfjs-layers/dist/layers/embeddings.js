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
 * TensorFlow.js Layers: Embedding Layer.
 *
 * Original source: keras/constraints.py
 */
import { notEqual, reshape, serialization, tidy, zerosLike } from '@tensorflow/tfjs-core';
import * as K from '../backend/tfjs_backend';
import { getConstraint, serializeConstraint } from '../constraints';
import { Layer } from '../engine/topology';
import { ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import * as generic_utils from '../utils/generic_utils';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
export class Embedding extends Layer {
    constructor(args) {
        super(args);
        this.embeddings = null;
        this.DEFAULT_EMBEDDINGS_INITIALIZER = 'randomUniform';
        if (args.batchInputShape == null && args.inputShape == null) {
            // Porting Note: This logic is copied from Layer's constructor, since we
            // can't do exactly what the Python constructor does for Embedding().
            // Specifically, the super constructor can not be called after the
            // mutation of the `config` argument.
            let batchSize = null;
            if (args.batchSize != null) {
                batchSize = args.batchSize;
            }
            if (args.inputLength == null) {
                // Fix super-constructor to what it would have done if
                // 'config.inputShape' were (None, )
                this.batchInputShape = [batchSize, null];
            }
            else {
                // Fix super-constructor to what it would have done if
                // 'config.inputShape' were (config.inputLength, )
                this.batchInputShape =
                    [batchSize].concat(generic_utils.toList(args.inputLength));
            }
        }
        this.inputDim = args.inputDim;
        generic_utils.assertPositiveInteger(this.inputDim, 'inputDim');
        this.outputDim = args.outputDim;
        generic_utils.assertPositiveInteger(this.outputDim, 'outputDim');
        this.embeddingsInitializer = getInitializer(args.embeddingsInitializer || this.DEFAULT_EMBEDDINGS_INITIALIZER);
        this.embeddingsRegularizer = getRegularizer(args.embeddingsRegularizer);
        this.activityRegularizer = getRegularizer(args.activityRegularizer);
        this.embeddingsConstraint = getConstraint(args.embeddingsConstraint);
        this.maskZero = args.maskZero;
        this.supportsMasking = args.maskZero;
        this.inputLength = args.inputLength;
    }
    build(inputShape) {
        this.embeddings = this.addWeight('embeddings', [this.inputDim, this.outputDim], this.dtype, this.embeddingsInitializer, this.embeddingsRegularizer, true, this.embeddingsConstraint);
        this.built = true;
    }
    // Override warnOnIncompatibleInputShape because an embedding layer allows
    // the input to have varying ranks.
    warnOnIncompatibleInputShape(inputShape) { }
    computeMask(inputs, mask) {
        return tidy(() => {
            if (!this.maskZero) {
                return null;
            }
            else {
                inputs = getExactlyOneTensor(inputs);
                return notEqual(inputs, zerosLike(inputs));
            }
        });
    }
    computeOutputShape(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        if (this.inputLength == null) {
            return [...inputShape, this.outputDim];
        }
        // inputLength can be an array if input is 3D or higher.
        const inLens = generic_utils.toList(this.inputLength);
        if (inLens.length !== inputShape.length - 1) {
            throw new ValueError(`"inputLength" is ${this.inputLength}, but received ` +
                `input shape has shape ${inputShape}`);
        }
        else {
            let i = 0;
            for (let k = 0; k < inLens.length; ++k) {
                const s1 = inLens[k];
                const s2 = inputShape[k + 1];
                if ((s1 != null) && (s2 != null) && (s1 !== s2)) {
                    throw new ValueError(`"inputLength" is ${this.inputLength}, but received ` +
                        `input shape has shape ${inputShape}`);
                }
                else if (s1 == null) {
                    inLens[i] = s2;
                }
                i++;
            }
        }
        return [inputShape[0], ...inLens, this.outputDim];
    }
    call(inputs, kwargs) {
        return tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            // Embedding layer accepts only a single input.
            let input = getExactlyOneTensor(inputs);
            if (input.dtype !== 'int32') {
                input = K.cast(input, 'int32');
            }
            const output = K.gather(this.embeddings.read(), reshape(input, [input.size]));
            return reshape(output, getExactlyOneShape(this.computeOutputShape(input.shape)));
        });
    }
    getConfig() {
        const config = {
            inputDim: this.inputDim,
            outputDim: this.outputDim,
            embeddingsInitializer: serializeInitializer(this.embeddingsInitializer),
            embeddingsRegularizer: serializeRegularizer(this.embeddingsRegularizer),
            activityRegularizer: serializeRegularizer(this.activityRegularizer),
            embeddingsConstraint: serializeConstraint(this.embeddingsConstraint),
            maskZero: this.maskZero,
            inputLength: this.inputLength
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
Embedding.className = 'Embedding';
serialization.registerClass(Embedding);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW1iZWRkaW5ncy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvZW1iZWRkaW5ncy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVIOzs7O0dBSUc7QUFDSCxPQUFPLEVBQUMsUUFBUSxFQUFFLE9BQU8sRUFBRSxhQUFhLEVBQVUsSUFBSSxFQUFFLFNBQVMsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRWhHLE9BQU8sS0FBSyxDQUFDLE1BQU0seUJBQXlCLENBQUM7QUFDN0MsT0FBTyxFQUFtQyxhQUFhLEVBQUUsbUJBQW1CLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUNwRyxPQUFPLEVBQUMsS0FBSyxFQUFZLE1BQU0sb0JBQW9CLENBQUM7QUFDcEQsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNyQyxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBRXpHLE9BQU8sRUFBQyxjQUFjLEVBQXNDLG9CQUFvQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFekcsT0FBTyxLQUFLLGFBQWEsTUFBTSx3QkFBd0IsQ0FBQztBQUN4RCxPQUFPLEVBQUMsa0JBQWtCLEVBQUUsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQWlEN0UsTUFBTSxPQUFPLFNBQVUsU0FBUSxLQUFLO0lBZ0JsQyxZQUFZLElBQXdCO1FBQ2xDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQztRQVJOLGVBQVUsR0FBa0IsSUFBSSxDQUFDO1FBRWhDLG1DQUE4QixHQUNuQyxlQUFlLENBQUM7UUFNbEIsSUFBSSxJQUFJLENBQUMsZUFBZSxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsVUFBVSxJQUFJLElBQUksRUFBRTtZQUMzRCx3RUFBd0U7WUFDeEUscUVBQXFFO1lBQ3JFLGtFQUFrRTtZQUNsRSxxQ0FBcUM7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QUFDO1FBQ0YsTUFBTSxVQUFVLEdBQUcsS0FBSyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ3JDLE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1FBQ2xDLE9BQU8sTUFBTSxDQUFDO0lBQ2hCLENBQUM7O0FBcklELGtCQUFrQjtBQUNYLG1CQUFTLEdBQUcsV0FBVyxDQUFDO0FBc0lqQyxhQUFhLENBQUMsYUFBYSxDQUFDLFNBQVMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQ1xuICpcbiAqIFVzZSBvZiB0aGlzIHNvdXJjZSBjb2RlIGlzIGdvdmVybmVkIGJ5IGFuIE1JVC1zdHlsZVxuICogbGljZW5zZSB0aGF0IGNhbiBiZSBmb3VuZCBpbiB0aGUgTElDRU5TRSBmaWxlIG9yIGF0XG4gKiBodHRwczovL29wZW5zb3VyY2Uub3JnL2xpY2Vuc2VzL01JVC5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBUZW5zb3JGbG93LmpzIExheWVyczogRW1iZWRkaW5nIExheWVyLlxuICpcbiAqIE9yaWdpbmFsIHNvdXJjZToga2VyYXMvY29uc3RyYWludHMucHlcbiAqL1xuaW1wb3J0IHtub3RFcXVhbCwgcmVzaGFwZSwgc2VyaWFsaXphdGlvbiwgVGVuc29yLCB0aWR5LCB6ZXJvc0xpa2V9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCAqIGFzIEsgZnJvbSAnLi4vYmFja2VuZC90ZmpzX2JhY2tlbmQnO1xuaW1wb3J0IHtDb25zdHJhaW50LCBDb25zdHJhaW50SWRlbnRpZmllciwgZ2V0Q29uc3RyYWludCwgc2VyaWFsaXplQ29uc3RyYWludH0gZnJvbSAnLi4vY29uc3RyYWludHMnO1xuaW1wb3J0IHtMYXllciwgTGF5ZXJBcmdzfSBmcm9tICcuLi9lbmdpbmUvdG9wb2xvZ3knO1xuaW1wb3J0IHtWYWx1ZUVycm9yfSBmcm9tICcuLi9lcnJvcnMnO1xuaW1wb3J0IHtnZXRJbml0aWFsaXplciwgSW5pdGlhbGl6ZXIsIEluaXRpYWxpemVySWRlbnRpZmllciwgc2VyaWFsaXplSW5pdGlhbGl6ZXJ9IGZyb20gJy4uL2luaXRpYWxpemVycyc7XG5pbXBvcnQge1NoYXBlfSBmcm9tICcuLi9rZXJhc19mb3JtYXQvY29tbW9uJztcbmltcG9ydCB7Z2V0UmVndWxhcml6ZXIsIFJlZ3VsYXJpemVyLCBSZWd1bGFyaXplcklkZW50aWZpZXIsIHNlcmlhbGl6ZVJlZ3VsYXJpemVyfSBmcm9tICcuLi9yZWd1bGFyaXplcnMnO1xuaW1wb3J0IHtLd2FyZ3N9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIGdlbmVyaWNfdXRpbHMgZnJvbSAnLi4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5pbXBvcnQge2dldEV4YWN0bHlPbmVTaGFwZSwgZ2V0RXhhY3RseU9uZVRlbnNvcn0gZnJvbSAnLi4vdXRpbHMvdHlwZXNfdXRpbHMnO1xuaW1wb3J0IHtMYXllclZhcmlhYmxlfSBmcm9tICcuLi92YXJpYWJsZXMnO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgRW1iZWRkaW5nTGF5ZXJBcmdzIGV4dGVuZHMgTGF5ZXJBcmdzIHtcbiAgLyoqXG4gICAqIEludGVnZXIgPiAwLiBTaXplIG9mIHRoZSB2b2NhYnVsYXJ5LCBpLmUuIG1heGltdW0gaW50ZWdlciBpbmRleCArIDEuXG4gICAqL1xuICBpbnB1dERpbTogbnVtYmVyO1xuICAvKipcbiAgICogSW50ZWdlciA+PSAwLiBEaW1lbnNpb24gb2YgdGhlIGRlbnNlIGVtYmVkZGluZy5cbiAgICovXG4gIG91dHB1dERpbTogbnVtYmVyO1xuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBgZW1iZWRkaW5nc2AgbWF0cml4LlxuICAgKi9cbiAgZW1iZWRkaW5nc0luaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJJZGVudGlmaWVyfEluaXRpYWxpemVyO1xuICAvKipcbiAgICogUmVndWxhcml6ZXIgZnVuY3Rpb24gYXBwbGllZCB0byB0aGUgYGVtYmVkZGluZ3NgIG1hdHJpeC5cbiAgICovXG4gIGVtYmVkZGluZ3NSZWd1bGFyaXplcj86IFJlZ3VsYXJpemVySWRlbnRpZmllcnxSZWd1bGFyaXplcjtcbiAgLyoqXG4gICAqIFJlZ3VsYXJpemVyIGZ1bmN0aW9uIGFwcGxpZWQgdG8gdGhlIGFjdGl2YXRpb24uXG4gICAqL1xuICBhY3Rpdml0eVJlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJJZGVudGlmaWVyfFJlZ3VsYXJpemVyO1xuICAvKipcbiAgICogQ29uc3RyYWludCBmdW5jdGlvbiBhcHBsaWVkIHRvIHRoZSBgZW1iZWRkaW5nc2AgbWF0cml4LlxuICAgKi9cbiAgZW1iZWRkaW5nc0NvbnN0cmFpbnQ/OiBDb25zdHJhaW50SWRlbnRpZmllcnxDb25zdHJhaW50O1xuICAvKipcbiAgICogV2hldGhlciB0aGUgaW5wdXQgdmFsdWUgMCBpcyBhIHNwZWNpYWwgXCJwYWRkaW5nXCIgdmFsdWUgdGhhdCBzaG91bGQgYmVcbiAgICogbWFza2VkIG91dC4gVGhpcyBpcyB1c2VmdWwgd2hlbiB1c2luZyByZWN1cnJlbnQgbGF5ZXJzIHdoaWNoIG1heSB0YWtlXG4gICAqIHZhcmlhYmxlIGxlbmd0aCBpbnB1dC5cbiAgICpcbiAgICogSWYgdGhpcyBpcyBgVHJ1ZWAgdGhlbiBhbGwgc3Vic2VxdWVudCBsYXllcnMgaW4gdGhlIG1vZGVsIG5lZWQgdG8gc3VwcG9ydFxuICAgKiBtYXNraW5nIG9yIGFuIGV4Y2VwdGlvbiB3aWxsIGJlIHJhaXNlZC4gSWYgbWFza1plcm8gaXMgc2V0IHRvIGBUcnVlYCwgYXMgYVxuICAgKiBjb25zZXF1ZW5jZSwgaW5kZXggMCBjYW5ub3QgYmUgdXNlZCBpbiB0aGUgdm9jYWJ1bGFyeSAoaW5wdXREaW0gc2hvdWxkXG4gICAqIGVxdWFsIHNpemUgb2Ygdm9jYWJ1bGFyeSArIDEpLlxuICAgKi9cbiAgbWFza1plcm8/OiBib29sZWFuO1xuICAvKipcbiAgICogTGVuZ3RoIG9mIGlucHV0IHNlcXVlbmNlcywgd2hlbiBpdCBpcyBjb25zdGFudC5cbiAgICpcbiAgICogVGhpcyBhcmd1bWVudCBpcyByZXF1aXJlZCBpZiB5b3UgYXJlIGdvaW5nIHRvIGNvbm5lY3QgYGZsYXR0ZW5gIHRoZW5cbiAgICogYGRlbnNlYCBsYXllcnMgdXBzdHJlYW0gKHdpdGhvdXQgaXQsIHRoZSBzaGFwZSBvZiB0aGUgZGVuc2Ugb3V0cHV0cyBjYW5ub3RcbiAgICogYmUgY29tcHV0ZWQpLlxuICAgKi9cbiAgaW5wdXRMZW5ndGg/OiBudW1iZXJ8bnVtYmVyW107XG59XG5cbmV4cG9ydCBjbGFzcyBFbWJlZGRpbmcgZXh0ZW5kcyBMYXllciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ0VtYmVkZGluZyc7XG4gIHByaXZhdGUgaW5wdXREaW06IG51bWJlcjtcbiAgcHJpdmF0ZSBvdXRwdXREaW06IG51bWJlcjtcbiAgcHJpdmF0ZSBlbWJlZGRpbmdzSW5pdGlhbGl6ZXI6IEluaXRpYWxpemVyO1xuICBwcml2YXRlIG1hc2taZXJvOiBib29sZWFuO1xuICBwcml2YXRlIGlucHV0TGVuZ3RoOiBudW1iZXJ8bnVtYmVyW107XG5cbiAgcHJpdmF0ZSBlbWJlZGRpbmdzOiBMYXllclZhcmlhYmxlID0gbnVsbDtcblxuICByZWFkb25seSBERUZBVUxUX0VNQkVERElOR1NfSU5JVElBTElaRVI6IEluaXRpYWxpemVySWRlbnRpZmllciA9XG4gICAgICAncmFuZG9tVW5pZm9ybSc7XG4gIHByaXZhdGUgcmVhZG9ubHkgZW1iZWRkaW5nc1JlZ3VsYXJpemVyPzogUmVndWxhcml6ZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgZW1iZWRkaW5nc0NvbnN0cmFpbnQ/OiBDb25zdHJhaW50O1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M6IEVtYmVkZGluZ0xheWVyQXJncykge1xuICAgIHN1cGVyKGFyZ3MpO1xuICAgIGlmIChhcmdzLmJhdGNoSW5wdXRTaGFwZSA9PSBudWxsICYmIGFyZ3MuaW5wdXRTaGFwZSA9PSBudWxsKSB7XG4gICAgICAvLyBQb3J0aW5nIE5vdGU6IFRoaXMgbG9naWMgaXMgY29waWVkIGZyb20gTGF5ZXIncyBjb25zdHJ1Y3Rvciwgc2luY2Ugd2VcbiAgICAgIC8vIGNhbid0IGRvIGV4YWN0bHkgd2hhdCB0aGUgUHl0aG9uIGNvbnN0cnVjdG9yIGRvZXMgZm9yIEVtYmVkZGluZygpLlxuICAgICAgLy8gU3BlY2lmaWNhbGx5LCB0aGUgc3VwZXIgY29uc3RydWN0b3IgY2FuIG5vdCBiZSBjYWxsZWQgYWZ0ZXIgdGhlXG4gICAgICAvLyBtdXRhdGlvbiBvZiB0aGUgYGNvbmZpZ2AgYXJndW1lbnQuXG4gICAgICBsZXQgYmF0Y2hTaXplOiBudW1iZXIgPSBudWxsO1xuICAgICAgaWYgKGFyZ3MuYmF0Y2hTaXplICE9IG51bGwpIHtcbiAgICAgICAgYmF0Y2hTaXplID0gYXJncy5iYXRjaFNpemU7XG4gICAgICB9XG4gICAgICBpZiAoYXJncy5pbnB1dExlbmd0aCA9PSBudWxsKSB7XG4gICAgICAgIC8vIEZpeCBzdXBlci1jb25zdHJ1Y3RvciB0byB3aGF0IGl0IHdvdWxkIGhhdmUgZG9uZSBpZlxuICAgICAgICAvLyAnY29uZmlnLmlucHV0U2hhcGUnIHdlcmUgKE5vbmUsIClcbiAgICAgICAgdGhpcy5iYXRjaElucHV0U2hhcGUgPSBbYmF0Y2hTaXplLCBudWxsXTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIEZpeCBzdXBlci1jb25zdHJ1Y3RvciB0byB3aGF0IGl0IHdvdWxkIGhhdmUgZG9uZSBpZlxuICAgICAgICAvLyAnY29uZmlnLmlucHV0U2hhcGUnIHdlcmUgKGNvbmZpZy5pbnB1dExlbmd0aCwgKVxuICAgICAgICB0aGlzLmJhdGNoSW5wdXRTaGFwZSA9XG4gICAgICAgICAgICBbYmF0Y2hTaXplXS5jb25jYXQoZ2VuZXJpY191dGlscy50b0xpc3QoYXJncy5pbnB1dExlbmd0aCkpO1xuICAgICAgfVxuICAgIH1cbiAgICB0aGlzLmlucHV0RGltID0gYXJncy5pbnB1dERpbTtcbiAgICBnZW5lcmljX3V0aWxzLmFzc2VydFBvc2l0aXZlSW50ZWdlcih0aGlzLmlucHV0RGltLCAnaW5wdXREaW0nKTtcbiAgICB0aGlzLm91dHB1dERpbSA9IGFyZ3Mub3V0cHV0RGltO1xuICAgIGdlbmVyaWNfdXRpbHMuYXNzZXJ0UG9zaXRpdmVJbnRlZ2VyKHRoaXMub3V0cHV0RGltLCAnb3V0cHV0RGltJyk7XG4gICAgdGhpcy5lbWJlZGRpbmdzSW5pdGlhbGl6ZXIgPSBnZXRJbml0aWFsaXplcihcbiAgICAgICAgYXJncy5lbWJlZGRpbmdzSW5pdGlhbGl6ZXIgfHwgdGhpcy5ERUZBVUxUX0VNQkVERElOR1NfSU5JVElBTElaRVIpO1xuICAgIHRoaXMuZW1iZWRkaW5nc1JlZ3VsYXJpemVyID0gZ2V0UmVndWxhcml6ZXIoYXJncy5lbWJlZGRpbmdzUmVndWxhcml6ZXIpO1xuICAgIHRoaXMuYWN0aXZpdHlSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYWN0aXZpdHlSZWd1bGFyaXplcik7XG4gICAgdGhpcy5lbWJlZGRpbmdzQ29uc3RyYWludCA9IGdldENvbnN0cmFpbnQoYXJncy5lbWJlZGRpbmdzQ29uc3RyYWludCk7XG4gICAgdGhpcy5tYXNrWmVybyA9IGFyZ3MubWFza1plcm87XG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSBhcmdzLm1hc2taZXJvO1xuICAgIHRoaXMuaW5wdXRMZW5ndGggPSBhcmdzLmlucHV0TGVuZ3RoO1xuICB9XG5cbiAgcHVibGljIG92ZXJyaWRlIGJ1aWxkKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiB2b2lkIHtcbiAgICB0aGlzLmVtYmVkZGluZ3MgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ2VtYmVkZGluZ3MnLCBbdGhpcy5pbnB1dERpbSwgdGhpcy5vdXRwdXREaW1dLCB0aGlzLmR0eXBlLFxuICAgICAgICB0aGlzLmVtYmVkZGluZ3NJbml0aWFsaXplciwgdGhpcy5lbWJlZGRpbmdzUmVndWxhcml6ZXIsIHRydWUsXG4gICAgICAgIHRoaXMuZW1iZWRkaW5nc0NvbnN0cmFpbnQpO1xuICAgIHRoaXMuYnVpbHQgPSB0cnVlO1xuICB9XG5cbiAgLy8gT3ZlcnJpZGUgd2Fybk9uSW5jb21wYXRpYmxlSW5wdXRTaGFwZSBiZWNhdXNlIGFuIGVtYmVkZGluZyBsYXllciBhbGxvd3NcbiAgLy8gdGhlIGlucHV0IHRvIGhhdmUgdmFyeWluZyByYW5rcy5cbiAgcHJvdGVjdGVkIG92ZXJyaWRlIHdhcm5PbkluY29tcGF0aWJsZUlucHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGUpIHt9XG5cbiAgb3ZlcnJpZGUgY29tcHV0ZU1hc2soaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIG1hc2s/OiBUZW5zb3J8VGVuc29yW10pOlxuICAgICAgVGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpZiAoIXRoaXMubWFza1plcm8pIHtcbiAgICAgICAgcmV0dXJuIG51bGw7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBpbnB1dHMgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICAgIHJldHVybiBub3RFcXVhbChpbnB1dHMsIHplcm9zTGlrZShpbnB1dHMpKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgaW5wdXRTaGFwZSA9IGdldEV4YWN0bHlPbmVTaGFwZShpbnB1dFNoYXBlKTtcbiAgICBpZiAodGhpcy5pbnB1dExlbmd0aCA9PSBudWxsKSB7XG4gICAgICByZXR1cm4gWy4uLmlucHV0U2hhcGUsIHRoaXMub3V0cHV0RGltXTtcbiAgICB9XG4gICAgLy8gaW5wdXRMZW5ndGggY2FuIGJlIGFuIGFycmF5IGlmIGlucHV0IGlzIDNEIG9yIGhpZ2hlci5cbiAgICBjb25zdCBpbkxlbnM6IG51bWJlcltdID0gZ2VuZXJpY191dGlscy50b0xpc3QodGhpcy5pbnB1dExlbmd0aCk7XG4gICAgaWYgKGluTGVucy5sZW5ndGggIT09IGlucHV0U2hhcGUubGVuZ3RoIC0gMSkge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYFwiaW5wdXRMZW5ndGhcIiBpcyAke3RoaXMuaW5wdXRMZW5ndGh9LCBidXQgcmVjZWl2ZWQgYCArXG4gICAgICAgICAgYGlucHV0IHNoYXBlIGhhcyBzaGFwZSAke2lucHV0U2hhcGV9YCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGxldCBpID0gMDtcbiAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgaW5MZW5zLmxlbmd0aDsgKytrKSB7XG4gICAgICAgIGNvbnN0IHMxID0gaW5MZW5zW2tdO1xuICAgICAgICBjb25zdCBzMiA9IGlucHV0U2hhcGVbayArIDFdO1xuICAgICAgICBpZiAoKHMxICE9IG51bGwpICYmIChzMiAhPSBudWxsKSAmJiAoczEgIT09IHMyKSkge1xuICAgICAgICAgIHRocm93IG5ldyBWYWx1ZUVycm9yKFxuICAgICAgICAgICAgICBgXCJpbnB1dExlbmd0aFwiIGlzICR7dGhpcy5pbnB1dExlbmd0aH0sIGJ1dCByZWNlaXZlZCBgICtcbiAgICAgICAgICAgICAgYGlucHV0IHNoYXBlIGhhcyBzaGFwZSAke2lucHV0U2hhcGV9YCk7XG4gICAgICAgIH0gZWxzZSBpZiAoczEgPT0gbnVsbCkge1xuICAgICAgICAgIGluTGVuc1tpXSA9IHMyO1xuICAgICAgICB9XG4gICAgICAgIGkrKztcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFtpbnB1dFNoYXBlWzBdLCAuLi5pbkxlbnMsIHRoaXMub3V0cHV0RGltXTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICB0aGlzLmludm9rZUNhbGxIb29rKGlucHV0cywga3dhcmdzKTtcbiAgICAgIC8vIEVtYmVkZGluZyBsYXllciBhY2NlcHRzIG9ubHkgYSBzaW5nbGUgaW5wdXQuXG4gICAgICBsZXQgaW5wdXQgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICBpZiAoaW5wdXQuZHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICAgICAgaW5wdXQgPSBLLmNhc3QoaW5wdXQsICdpbnQzMicpO1xuICAgICAgfVxuICAgICAgY29uc3Qgb3V0cHV0ID1cbiAgICAgICAgICBLLmdhdGhlcih0aGlzLmVtYmVkZGluZ3MucmVhZCgpLCByZXNoYXBlKGlucHV0LCBbaW5wdXQuc2l6ZV0pKTtcbiAgICAgIHJldHVybiByZXNoYXBlKFxuICAgICAgICAgIG91dHB1dCwgZ2V0RXhhY3RseU9uZVNoYXBlKHRoaXMuY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0LnNoYXBlKSkpO1xuICAgIH0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnID0ge1xuICAgICAgaW5wdXREaW06IHRoaXMuaW5wdXREaW0sXG4gICAgICBvdXRwdXREaW06IHRoaXMub3V0cHV0RGltLFxuICAgICAgZW1iZWRkaW5nc0luaXRpYWxpemVyOiBzZXJpYWxpemVJbml0aWFsaXplcih0aGlzLmVtYmVkZGluZ3NJbml0aWFsaXplciksXG4gICAgICBlbWJlZGRpbmdzUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuZW1iZWRkaW5nc1JlZ3VsYXJpemVyKSxcbiAgICAgIGFjdGl2aXR5UmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuYWN0aXZpdHlSZWd1bGFyaXplciksXG4gICAgICBlbWJlZGRpbmdzQ29uc3RyYWludDogc2VyaWFsaXplQ29uc3RyYWludCh0aGlzLmVtYmVkZGluZ3NDb25zdHJhaW50KSxcbiAgICAgIG1hc2taZXJvOiB0aGlzLm1hc2taZXJvLFxuICAgICAgaW5wdXRMZW5ndGg6IHRoaXMuaW5wdXRMZW5ndGhcbiAgICB9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEVtYmVkZGluZyk7XG4iXX0=