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
 *  Advanced activation layers.
 */
import { cast, clipByValue, elu, greater, leakyRelu, mul, prelu, relu, serialization } from '@tensorflow/tfjs-core';
import { Softmax as softmaxActivation } from '../activations';
import { getConstraint, serializeConstraint } from '../constraints';
import { InputSpec, Layer } from '../engine/topology';
import { NotImplementedError, ValueError } from '../errors';
import { getInitializer, serializeInitializer } from '../initializers';
import { getRegularizer, serializeRegularizer } from '../regularizers';
import { getExactlyOneShape, getExactlyOneTensor } from '../utils/types_utils';
export class ReLU extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.supportsMasking = true;
        if (args != null) {
            this.maxValue = args.maxValue;
        }
    }
    call(inputs, kwargs) {
        inputs = getExactlyOneTensor(inputs);
        let output = relu(inputs);
        if (this.maxValue != null) {
            output = clipByValue(output, 0, this.maxValue);
        }
        return output;
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const config = { maxValue: this.maxValue };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
ReLU.className = 'ReLU';
serialization.registerClass(ReLU);
export class LeakyReLU extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.DEFAULT_ALPHA = 0.3;
        if (args == null) {
            args = {};
        }
        this.alpha = args.alpha == null ? this.DEFAULT_ALPHA : args.alpha;
    }
    call(inputs, kwargs) {
        const x = getExactlyOneTensor(inputs);
        return leakyRelu(x, this.alpha);
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const config = { alpha: this.alpha };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
LeakyReLU.className = 'LeakyReLU';
serialization.registerClass(LeakyReLU);
export class PReLU extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.DEFAULT_ALPHA_INITIALIZER = 'zeros';
        if (args == null) {
            args = {};
        }
        this.supportsMasking = true;
        this.alphaInitializer =
            getInitializer(args.alphaInitializer || this.DEFAULT_ALPHA_INITIALIZER);
        this.alphaRegularizer = getRegularizer(args.alphaRegularizer);
        this.alphaConstraint = getConstraint(args.alphaConstraint);
        if (args.sharedAxes == null) {
            this.sharedAxes = null;
        }
        else if (Array.isArray(args.sharedAxes)) {
            this.sharedAxes = args.sharedAxes;
        }
        else if (typeof args.sharedAxes === 'number') {
            this.sharedAxes = [args.sharedAxes];
        }
        else {
            throw new ValueError(`Expected sharedAxes to be a number or an array of numbers, ` +
                `but got ${args.sharedAxes}`);
        }
    }
    build(inputShape) {
        inputShape = getExactlyOneShape(inputShape);
        const paramShape = inputShape.slice(1);
        if (this.sharedAxes != null) {
            for (const i of this.sharedAxes) {
                paramShape[i - 1] = 1;
            }
        }
        this.alpha = this.addWeight('alpha', paramShape, 'float32', this.alphaInitializer, this.alphaRegularizer, true, this.alphaConstraint);
        // Set input spec.
        const axes = {};
        if (this.sharedAxes != null) {
            for (let i = 1; i < inputShape.length; ++i) {
                axes[i] = inputShape[i];
            }
        }
        this.inputSpec = [new InputSpec({
                ndim: inputShape.length,
                axes,
            })];
        this.built = true;
    }
    call(inputs, kwargs) {
        inputs = getExactlyOneTensor(inputs);
        return prelu(inputs, this.alpha.read());
    }
    getConfig() {
        const config = {
            alphaInitializer: serializeInitializer(this.alphaInitializer),
            alphaRegularizer: serializeRegularizer(this.alphaRegularizer),
            alphaConstraint: serializeConstraint(this.alphaConstraint),
            sharedAxes: this.sharedAxes
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
PReLU.className = 'PReLU';
serialization.registerClass(PReLU);
export class ELU extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.DEFAULT_ALPHA = 1.0;
        if (args == null) {
            args = {};
        }
        if (args.alpha != null && args.alpha !== this.DEFAULT_ALPHA) {
            throw new NotImplementedError(`Non-default alpha value (${args.alpha}) is not supported by the ` +
                `ELU layer yet.`);
        }
        this.alpha = args.alpha == null ? this.DEFAULT_ALPHA : args.alpha;
    }
    call(inputs, kwargs) {
        const x = getExactlyOneTensor(inputs);
        return elu(x);
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const config = { alpha: this.alpha };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
ELU.className = 'ELU';
serialization.registerClass(ELU);
export class ThresholdedReLU extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.DEFAULT_THETA = 1.0;
        if (args == null) {
            args = {};
        }
        this.theta = args.theta == null ? this.DEFAULT_THETA : args.theta;
    }
    call(inputs, kwargs) {
        const x = getExactlyOneTensor(inputs);
        return mul(x, cast(greater(x, this.theta), 'float32'));
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const config = { theta: this.theta };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
ThresholdedReLU.className = 'ThresholdedReLU';
serialization.registerClass(ThresholdedReLU);
export class Softmax extends Layer {
    constructor(args) {
        super(args == null ? {} : args);
        this.DEFAULT_AXIS = 1.0;
        if (args == null) {
            args = {};
        }
        this.softmax = new softmaxActivation().apply;
        this.axis = args.axis == null ? this.DEFAULT_AXIS : args.axis;
    }
    call(inputs, kwargs) {
        const x = getExactlyOneTensor(inputs);
        return this.softmax(x, this.axis);
    }
    computeOutputShape(inputShape) {
        return inputShape;
    }
    getConfig() {
        const config = { axis: this.axis };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
}
/** @nocollapse */
Softmax.className = 'Softmax';
serialization.registerClass(Softmax);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWR2YW5jZWRfYWN0aXZhdGlvbnMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvbGF5ZXJzL2FkdmFuY2VkX2FjdGl2YXRpb25zLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUg7O0dBRUc7QUFFSCxPQUFPLEVBQUMsSUFBSSxFQUFFLFdBQVcsRUFBRSxHQUFHLEVBQUUsT0FBTyxFQUFFLFNBQVMsRUFBRSxHQUFHLEVBQUUsS0FBSyxFQUFFLElBQUksRUFBRSxhQUFhLEVBQVMsTUFBTSx1QkFBdUIsQ0FBQztBQUUxSCxPQUFPLEVBQUMsT0FBTyxJQUFJLGlCQUFpQixFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDNUQsT0FBTyxFQUFhLGFBQWEsRUFBRSxtQkFBbUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzlFLE9BQU8sRUFBQyxTQUFTLEVBQUUsS0FBSyxFQUFZLE1BQU0sb0JBQW9CLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFFLFVBQVUsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUMxRCxPQUFPLEVBQUMsY0FBYyxFQUFzQyxvQkFBb0IsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBRXpHLE9BQU8sRUFBQyxjQUFjLEVBQWUsb0JBQW9CLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUVsRixPQUFPLEVBQUMsa0JBQWtCLEVBQUUsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQVU3RSxNQUFNLE9BQU8sSUFBSyxTQUFRLEtBQUs7SUFLN0IsWUFBWSxJQUFvQjtRQUM5QixLQUFLLENBQUMsSUFBSSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNoQyxJQUFJLENBQUMsZUFBZSxHQUFHLElBQUksQ0FBQztRQUM1QixJQUFJLElBQUksSUFBSSxJQUFJLEVBQUU7WUFDaEIs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O1FBQzNELE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxNQUFNLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNsQyxPQUFPLE1BQU0sQ0FBQztJQUNoQixDQUFDOztBQTdCRCxrQkFBa0I7QUFDWCxpQkFBUyxHQUFHLFNBQVMsQ0FBQztBQThCL0IsYUFBYSxDQUFDLGFBQWEsQ0FBQyxPQUFPLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbi8qKlxuICogIEFkdmFuY2VkIGFjdGl2YXRpb24gbGF5ZXJzLlxuICovXG5cbmltcG9ydCB7Y2FzdCwgY2xpcEJ5VmFsdWUsIGVsdSwgZ3JlYXRlciwgbGVha3lSZWx1LCBtdWwsIHByZWx1LCByZWx1LCBzZXJpYWxpemF0aW9uLCBUZW5zb3J9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7U29mdG1heCBhcyBzb2Z0bWF4QWN0aXZhdGlvbn0gZnJvbSAnLi4vYWN0aXZhdGlvbnMnO1xuaW1wb3J0IHtDb25zdHJhaW50LCBnZXRDb25zdHJhaW50LCBzZXJpYWxpemVDb25zdHJhaW50fSBmcm9tICcuLi9jb25zdHJhaW50cyc7XG5pbXBvcnQge0lucHV0U3BlYywgTGF5ZXIsIExheWVyQXJnc30gZnJvbSAnLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7Tm90SW1wbGVtZW50ZWRFcnJvciwgVmFsdWVFcnJvcn0gZnJvbSAnLi4vZXJyb3JzJztcbmltcG9ydCB7Z2V0SW5pdGlhbGl6ZXIsIEluaXRpYWxpemVyLCBJbml0aWFsaXplcklkZW50aWZpZXIsIHNlcmlhbGl6ZUluaXRpYWxpemVyfSBmcm9tICcuLi9pbml0aWFsaXplcnMnO1xuaW1wb3J0IHtTaGFwZX0gZnJvbSAnLi4va2VyYXNfZm9ybWF0L2NvbW1vbic7XG5pbXBvcnQge2dldFJlZ3VsYXJpemVyLCBSZWd1bGFyaXplciwgc2VyaWFsaXplUmVndWxhcml6ZXJ9IGZyb20gJy4uL3JlZ3VsYXJpemVycyc7XG5pbXBvcnQge0t3YXJnc30gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtnZXRFeGFjdGx5T25lU2hhcGUsIGdldEV4YWN0bHlPbmVUZW5zb3J9IGZyb20gJy4uL3V0aWxzL3R5cGVzX3V0aWxzJztcbmltcG9ydCB7TGF5ZXJWYXJpYWJsZX0gZnJvbSAnLi4vdmFyaWFibGVzJztcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFJlTFVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRmxvYXQsIHRoZSBtYXhpbXVtIG91dHB1dCB2YWx1ZS5cbiAgICovXG4gIG1heFZhbHVlPzogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgUmVMVSBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnUmVMVSc7XG4gIG1heFZhbHVlOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3IoYXJncz86IFJlTFVMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzID09IG51bGwgPyB7fSA6IGFyZ3MpO1xuICAgIHRoaXMuc3VwcG9ydHNNYXNraW5nID0gdHJ1ZTtcbiAgICBpZiAoYXJncyAhPSBudWxsKSB7XG4gICAgICB0aGlzLm1heFZhbHVlID0gYXJncy5tYXhWYWx1ZTtcbiAgICB9XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgIGxldCBvdXRwdXQgPSByZWx1KGlucHV0cyk7XG4gICAgaWYgKHRoaXMubWF4VmFsdWUgIT0gbnVsbCkge1xuICAgICAgb3V0cHV0ID0gY2xpcEJ5VmFsdWUob3V0cHV0LCAwLCB0aGlzLm1heFZhbHVlKTtcbiAgICB9XG4gICAgcmV0dXJuIG91dHB1dDtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgcmV0dXJuIGlucHV0U2hhcGU7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHttYXhWYWx1ZTogdGhpcy5tYXhWYWx1ZX07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoUmVMVSk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBMZWFreVJlTFVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRmxvYXQgYD49IDBgLiBOZWdhdGl2ZSBzbG9wZSBjb2VmZmljaWVudC4gRGVmYXVsdHMgdG8gYDAuM2AuXG4gICAqL1xuICBhbHBoYT86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIExlYWt5UmVMVSBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTGVha3lSZUxVJztcbiAgcmVhZG9ubHkgYWxwaGE6IG51bWJlcjtcblxuICByZWFkb25seSBERUZBVUxUX0FMUEhBID0gMC4zO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBMZWFreVJlTFVMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzID09IG51bGwgPyB7fSA6IGFyZ3MpO1xuICAgIGlmIChhcmdzID09IG51bGwpIHtcbiAgICAgIGFyZ3MgPSB7fTtcbiAgICB9XG4gICAgdGhpcy5hbHBoYSA9IGFyZ3MuYWxwaGEgPT0gbnVsbCA/IHRoaXMuREVGQVVMVF9BTFBIQSA6IGFyZ3MuYWxwaGE7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgY29uc3QgeCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICByZXR1cm4gbGVha3lSZWx1KHgsIHRoaXMuYWxwaGEpO1xuICB9XG5cbiAgb3ZlcnJpZGUgY29tcHV0ZU91dHB1dFNoYXBlKGlucHV0U2hhcGU6IFNoYXBlfFNoYXBlW10pOiBTaGFwZXxTaGFwZVtdIHtcbiAgICByZXR1cm4gaW5wdXRTaGFwZTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge2FscGhhOiB0aGlzLmFscGhhfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhMZWFreVJlTFUpO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgUFJlTFVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogSW5pdGlhbGl6ZXIgZm9yIHRoZSBsZWFybmFibGUgYWxwaGEuXG4gICAqL1xuICBhbHBoYUluaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJ8SW5pdGlhbGl6ZXJJZGVudGlmaWVyO1xuXG4gIC8qKlxuICAgKiBSZWd1bGFyaXplciBmb3IgdGhlIGxlYXJuYWJsZSBhbHBoYS5cbiAgICovXG4gIGFscGhhUmVndWxhcml6ZXI/OiBSZWd1bGFyaXplcjtcblxuICAvKipcbiAgICogQ29uc3RyYWludCBmb3IgdGhlIGxlYXJuYWJsZSBhbHBoYS5cbiAgICovXG4gIGFscGhhQ29uc3RyYWludD86IENvbnN0cmFpbnQ7XG5cbiAgLyoqXG4gICAqIFRoZSBheGVzIGFsb25nIHdoaWNoIHRvIHNoYXJlIGxlYXJuYWJsZSBwYXJhbWV0ZXJzIGZvciB0aGUgYWN0aXZhdGlvblxuICAgKiBmdW5jdGlvbi4gRm9yIGV4YW1wbGUsIGlmIHRoZSBpbmNvbWluZyBmZWF0dXJlIG1hcHMgYXJlIGZyb20gYSAyRFxuICAgKiBjb252b2x1dGlvbiB3aXRoIG91dHB1dCBzaGFwZSBgW251bUV4YW1wbGVzLCBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc11gLFxuICAgKiBhbmQgeW91IHdpc2ggdG8gc2hhcmUgcGFyYW1ldGVycyBhY3Jvc3Mgc3BhY2UgKGhlaWdodCBhbmQgd2lkdGgpIHNvIHRoYXRcbiAgICogZWFjaCBmaWx0ZXIgY2hhbm5lbHMgaGFzIG9ubHkgb25lIHNldCBvZiBwYXJhbWV0ZXJzLCBzZXRcbiAgICogYHNoYXJlZF9heGVzOiBbMSwgMl1gLlxuICAgKi9cbiAgc2hhcmVkQXhlcz86IG51bWJlcnxudW1iZXJbXTtcbn1cblxuZXhwb3J0IGNsYXNzIFBSZUxVIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdQUmVMVSc7XG4gIHByaXZhdGUgcmVhZG9ubHkgYWxwaGFJbml0aWFsaXplcjogSW5pdGlhbGl6ZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgYWxwaGFSZWd1bGFyaXplcjogUmVndWxhcml6ZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgYWxwaGFDb25zdHJhaW50OiBDb25zdHJhaW50O1xuICBwcml2YXRlIHJlYWRvbmx5IHNoYXJlZEF4ZXM6IG51bWJlcltdO1xuICBwcml2YXRlIGFscGhhOiBMYXllclZhcmlhYmxlO1xuXG4gIHJlYWRvbmx5IERFRkFVTFRfQUxQSEFfSU5JVElBTElaRVI6IEluaXRpYWxpemVySWRlbnRpZmllciA9ICd6ZXJvcyc7XG5cbiAgY29uc3RydWN0b3IoYXJncz86IFBSZUxVTGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyA9PSBudWxsID8ge30gOiBhcmdzKTtcbiAgICBpZiAoYXJncyA9PSBudWxsKSB7XG4gICAgICBhcmdzID0ge307XG4gICAgfVxuXG4gICAgdGhpcy5zdXBwb3J0c01hc2tpbmcgPSB0cnVlO1xuICAgIHRoaXMuYWxwaGFJbml0aWFsaXplciA9XG4gICAgICAgIGdldEluaXRpYWxpemVyKGFyZ3MuYWxwaGFJbml0aWFsaXplciB8fCB0aGlzLkRFRkFVTFRfQUxQSEFfSU5JVElBTElaRVIpO1xuICAgIHRoaXMuYWxwaGFSZWd1bGFyaXplciA9IGdldFJlZ3VsYXJpemVyKGFyZ3MuYWxwaGFSZWd1bGFyaXplcik7XG4gICAgdGhpcy5hbHBoYUNvbnN0cmFpbnQgPSBnZXRDb25zdHJhaW50KGFyZ3MuYWxwaGFDb25zdHJhaW50KTtcbiAgICBpZiAoYXJncy5zaGFyZWRBeGVzID09IG51bGwpIHtcbiAgICAgIHRoaXMuc2hhcmVkQXhlcyA9IG51bGw7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KGFyZ3Muc2hhcmVkQXhlcykpIHtcbiAgICAgIHRoaXMuc2hhcmVkQXhlcyA9IGFyZ3Muc2hhcmVkQXhlcztcbiAgICB9IGVsc2UgaWYgKHR5cGVvZiBhcmdzLnNoYXJlZEF4ZXMgPT09ICdudW1iZXInKSB7XG4gICAgICB0aGlzLnNoYXJlZEF4ZXMgPSBbYXJncy5zaGFyZWRBeGVzXTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IFZhbHVlRXJyb3IoXG4gICAgICAgICAgYEV4cGVjdGVkIHNoYXJlZEF4ZXMgdG8gYmUgYSBudW1iZXIgb3IgYW4gYXJyYXkgb2YgbnVtYmVycywgYCArXG4gICAgICAgICAgYGJ1dCBnb3QgJHthcmdzLnNoYXJlZEF4ZXN9YCk7XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgYnVpbGQoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSkge1xuICAgIGlucHV0U2hhcGUgPSBnZXRFeGFjdGx5T25lU2hhcGUoaW5wdXRTaGFwZSk7XG4gICAgY29uc3QgcGFyYW1TaGFwZTogU2hhcGUgPSBpbnB1dFNoYXBlLnNsaWNlKDEpO1xuICAgIGlmICh0aGlzLnNoYXJlZEF4ZXMgIT0gbnVsbCkge1xuICAgICAgZm9yIChjb25zdCBpIG9mIHRoaXMuc2hhcmVkQXhlcykge1xuICAgICAgICBwYXJhbVNoYXBlW2kgLSAxXSA9IDE7XG4gICAgICB9XG4gICAgfVxuICAgIHRoaXMuYWxwaGEgPSB0aGlzLmFkZFdlaWdodChcbiAgICAgICAgJ2FscGhhJywgcGFyYW1TaGFwZSwgJ2Zsb2F0MzInLCB0aGlzLmFscGhhSW5pdGlhbGl6ZXIsXG4gICAgICAgIHRoaXMuYWxwaGFSZWd1bGFyaXplciwgdHJ1ZSwgdGhpcy5hbHBoYUNvbnN0cmFpbnQpO1xuICAgIC8vIFNldCBpbnB1dCBzcGVjLlxuICAgIGNvbnN0IGF4ZXM6IHtbYXhpczogbnVtYmVyXTogbnVtYmVyfSA9IHt9O1xuICAgIGlmICh0aGlzLnNoYXJlZEF4ZXMgIT0gbnVsbCkge1xuICAgICAgZm9yIChsZXQgaSA9IDE7IGkgPCBpbnB1dFNoYXBlLmxlbmd0aDsgKytpKSB7XG4gICAgICAgIGF4ZXNbaV0gPSBpbnB1dFNoYXBlW2ldO1xuICAgICAgfVxuICAgIH1cbiAgICB0aGlzLmlucHV0U3BlYyA9IFtuZXcgSW5wdXRTcGVjKHtcbiAgICAgIG5kaW06IGlucHV0U2hhcGUubGVuZ3RoLFxuICAgICAgYXhlcyxcbiAgICB9KV07XG4gICAgdGhpcy5idWlsdCA9IHRydWU7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgaW5wdXRzID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgIHJldHVybiBwcmVsdShpbnB1dHMsIHRoaXMuYWxwaGEucmVhZCgpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGdldENvbmZpZygpOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3Qge1xuICAgIGNvbnN0IGNvbmZpZzogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0ID0ge1xuICAgICAgYWxwaGFJbml0aWFsaXplcjogc2VyaWFsaXplSW5pdGlhbGl6ZXIodGhpcy5hbHBoYUluaXRpYWxpemVyKSxcbiAgICAgIGFscGhhUmVndWxhcml6ZXI6IHNlcmlhbGl6ZVJlZ3VsYXJpemVyKHRoaXMuYWxwaGFSZWd1bGFyaXplciksXG4gICAgICBhbHBoYUNvbnN0cmFpbnQ6IHNlcmlhbGl6ZUNvbnN0cmFpbnQodGhpcy5hbHBoYUNvbnN0cmFpbnQpLFxuICAgICAgc2hhcmVkQXhlczogdGhpcy5zaGFyZWRBeGVzXG4gICAgfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhQUmVMVSk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBFTFVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRmxvYXQgYD49IDBgLiBOZWdhdGl2ZSBzbG9wZSBjb2VmZmljaWVudC4gRGVmYXVsdHMgdG8gYDEuMGAuXG4gICAqL1xuICBhbHBoYT86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIEVMVSBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnRUxVJztcbiAgcmVhZG9ubHkgYWxwaGE6IG51bWJlcjtcblxuICByZWFkb25seSBERUZBVUxUX0FMUEhBID0gMS4wO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBFTFVMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzID09IG51bGwgPyB7fSA6IGFyZ3MpO1xuICAgIGlmIChhcmdzID09IG51bGwpIHtcbiAgICAgIGFyZ3MgPSB7fTtcbiAgICB9XG5cbiAgICBpZiAoYXJncy5hbHBoYSAhPSBudWxsICYmIGFyZ3MuYWxwaGEgIT09IHRoaXMuREVGQVVMVF9BTFBIQSkge1xuICAgICAgdGhyb3cgbmV3IE5vdEltcGxlbWVudGVkRXJyb3IoXG4gICAgICAgICAgYE5vbi1kZWZhdWx0IGFscGhhIHZhbHVlICgke2FyZ3MuYWxwaGF9KSBpcyBub3Qgc3VwcG9ydGVkIGJ5IHRoZSBgICtcbiAgICAgICAgICBgRUxVIGxheWVyIHlldC5gKTtcbiAgICB9XG5cbiAgICB0aGlzLmFscGhhID0gYXJncy5hbHBoYSA9PSBudWxsID8gdGhpcy5ERUZBVUxUX0FMUEhBIDogYXJncy5hbHBoYTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICBjb25zdCB4ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgIHJldHVybiBlbHUoeCk7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIHJldHVybiBpbnB1dFNoYXBlO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7YWxwaGE6IHRoaXMuYWxwaGF9O1xuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxufVxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKEVMVSk7XG5cbmV4cG9ydCBkZWNsYXJlIGludGVyZmFjZSBUaHJlc2hvbGRlZFJlTFVMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogRmxvYXQgPj0gMC4gVGhyZXNob2xkIGxvY2F0aW9uIG9mIGFjdGl2YXRpb24uXG4gICAqL1xuICB0aGV0YT86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIFRocmVzaG9sZGVkUmVMVSBleHRlbmRzIExheWVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnVGhyZXNob2xkZWRSZUxVJztcbiAgcmVhZG9ubHkgdGhldGE6IG51bWJlcjtcblxuICByZWFkb25seSBERUZBVUxUX1RIRVRBID0gMS4wO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBUaHJlc2hvbGRlZFJlTFVMYXllckFyZ3MpIHtcbiAgICBzdXBlcihhcmdzID09IG51bGwgPyB7fSA6IGFyZ3MpO1xuICAgIGlmIChhcmdzID09IG51bGwpIHtcbiAgICAgIGFyZ3MgPSB7fTtcbiAgICB9XG5cbiAgICB0aGlzLnRoZXRhID0gYXJncy50aGV0YSA9PSBudWxsID8gdGhpcy5ERUZBVUxUX1RIRVRBIDogYXJncy50aGV0YTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yfFRlbnNvcltdIHtcbiAgICBjb25zdCB4ID0gZ2V0RXhhY3RseU9uZVRlbnNvcihpbnB1dHMpO1xuICAgIHJldHVybiBtdWwoeCwgY2FzdChncmVhdGVyKHgsIHRoaXMudGhldGEpLCAnZmxvYXQzMicpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGNvbXB1dGVPdXRwdXRTaGFwZShpbnB1dFNoYXBlOiBTaGFwZXxTaGFwZVtdKTogU2hhcGV8U2hhcGVbXSB7XG4gICAgcmV0dXJuIGlucHV0U2hhcGU7XG4gIH1cblxuICBvdmVycmlkZSBnZXRDb25maWcoKTogc2VyaWFsaXphdGlvbi5Db25maWdEaWN0IHtcbiAgICBjb25zdCBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCA9IHt0aGV0YTogdGhpcy50aGV0YX07XG4gICAgY29uc3QgYmFzZUNvbmZpZyA9IHN1cGVyLmdldENvbmZpZygpO1xuICAgIE9iamVjdC5hc3NpZ24oY29uZmlnLCBiYXNlQ29uZmlnKTtcbiAgICByZXR1cm4gY29uZmlnO1xuICB9XG59XG5zZXJpYWxpemF0aW9uLnJlZ2lzdGVyQ2xhc3MoVGhyZXNob2xkZWRSZUxVKTtcblxuZXhwb3J0IGRlY2xhcmUgaW50ZXJmYWNlIFNvZnRtYXhMYXllckFyZ3MgZXh0ZW5kcyBMYXllckFyZ3Mge1xuICAvKipcbiAgICogSW50ZWdlciwgYXhpcyBhbG9uZyB3aGljaCB0aGUgc29mdG1heCBub3JtYWxpemF0aW9uIGlzIGFwcGxpZWQuXG4gICAqIERlZmF1bHRzIHRvIGAtMWAgKGkuZS4sIHRoZSBsYXN0IGF4aXMpLlxuICAgKi9cbiAgYXhpcz86IG51bWJlcjtcbn1cblxuZXhwb3J0IGNsYXNzIFNvZnRtYXggZXh0ZW5kcyBMYXllciB7XG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgY2xhc3NOYW1lID0gJ1NvZnRtYXgnO1xuICByZWFkb25seSBheGlzOiBudW1iZXI7XG4gIHJlYWRvbmx5IHNvZnRtYXg6ICh0OiBUZW5zb3IsIGE/OiBudW1iZXIpID0+IFRlbnNvcjtcbiAgcmVhZG9ubHkgREVGQVVMVF9BWElTID0gMS4wO1xuXG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBTb2Z0bWF4TGF5ZXJBcmdzKSB7XG4gICAgc3VwZXIoYXJncyA9PSBudWxsID8ge30gOiBhcmdzKTtcbiAgICBpZiAoYXJncyA9PSBudWxsKSB7XG4gICAgICBhcmdzID0ge307XG4gICAgfVxuICAgIHRoaXMuc29mdG1heCA9IG5ldyBzb2Z0bWF4QWN0aXZhdGlvbigpLmFwcGx5O1xuICAgIHRoaXMuYXhpcyA9IGFyZ3MuYXhpcyA9PSBudWxsID8gdGhpcy5ERUZBVUxUX0FYSVMgOiBhcmdzLmF4aXM7XG4gIH1cblxuICBvdmVycmlkZSBjYWxsKGlucHV0czogVGVuc29yfFRlbnNvcltdLCBrd2FyZ3M6IEt3YXJncyk6IFRlbnNvcnxUZW5zb3JbXSB7XG4gICAgY29uc3QgeCA9IGdldEV4YWN0bHlPbmVUZW5zb3IoaW5wdXRzKTtcbiAgICByZXR1cm4gdGhpcy5zb2Z0bWF4KHgsIHRoaXMuYXhpcyk7XG4gIH1cblxuICBvdmVycmlkZSBjb21wdXRlT3V0cHV0U2hhcGUoaW5wdXRTaGFwZTogU2hhcGV8U2hhcGVbXSk6IFNoYXBlfFNoYXBlW10ge1xuICAgIHJldHVybiBpbnB1dFNoYXBlO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7YXhpczogdGhpcy5heGlzfTtcbiAgICBjb25zdCBiYXNlQ29uZmlnID0gc3VwZXIuZ2V0Q29uZmlnKCk7XG4gICAgT2JqZWN0LmFzc2lnbihjb25maWcsIGJhc2VDb25maWcpO1xuICAgIHJldHVybiBjb25maWc7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhTb2Z0bWF4KTtcbiJdfQ==