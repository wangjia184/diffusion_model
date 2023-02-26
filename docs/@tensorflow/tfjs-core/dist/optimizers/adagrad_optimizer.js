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
import { ENGINE } from '../engine';
import { dispose, tidy } from '../globals';
import { add } from '../ops/add';
import { div } from '../ops/div';
import { fill } from '../ops/fill';
import { mul } from '../ops/mul';
import { sqrt } from '../ops/sqrt';
import { square } from '../ops/square';
import { Optimizer } from './optimizer';
/** @doclink Optimizer */
export class AdagradOptimizer extends Optimizer {
    constructor(learningRate, initialAccumulatorValue = 0.1) {
        super();
        this.learningRate = learningRate;
        this.initialAccumulatorValue = initialAccumulatorValue;
        this.accumulatedGrads = [];
    }
    /** @nocollapse */
    static get className() {
        // Name matters for Python compatibility.
        // This is a getter instead of a property because when it's a property, it
        // prevents the entire class from being tree-shaken.
        return 'Adagrad';
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            if (this.accumulatedGrads[i] == null) {
                const trainable = false;
                this.accumulatedGrads[i] = {
                    originalName: `${name}/accumulator`,
                    variable: tidy(() => fill(value.shape, this.initialAccumulatorValue)
                        .variable(trainable))
                };
            }
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            const accumulatedGrad = this.accumulatedGrads[i].variable;
            tidy(() => {
                const newAccumulatedGrad = add(accumulatedGrad, square(gradient));
                accumulatedGrad.assign(newAccumulatedGrad);
                const newValue = add(mul(div(gradient, sqrt(add(newAccumulatedGrad, ENGINE.backend.epsilon()))), -this.learningRate), value);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        if (this.accumulatedGrads != null) {
            dispose(this.accumulatedGrads.map(v => v.variable));
        }
    }
    async getWeights() {
        // Order matters for Python compatibility.
        return [await this.saveIterations()].concat(this.accumulatedGrads.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const trainable = false;
        this.accumulatedGrads = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'initialAccumulatorValue': this.initialAccumulatorValue,
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['initialAccumulatorValue']);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYWRhZ3JhZF9vcHRpbWl6ZXIuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wdGltaXplcnMvYWRhZ3JhZF9vcHRpbWl6ZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsT0FBTyxFQUFFLElBQUksRUFBQyxNQUFNLFlBQVksQ0FBQztBQUN6QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDL0IsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUlyQyxPQUFPLEVBQUMsU0FBUyxFQUFvQixNQUFNLGFBQWEsQ0FBQztBQUV6RCx5QkFBeUI7QUFDekIsTUFBTSxPQUFPLGdCQUFpQixTQUFRLFNBQVM7SUFXN0MsWUFDYyxZQUFvQixFQUFVLDBCQUEwQixHQUFHO1FBQ3ZFLEtBQUssRUFBRSxDQUFDO1FBREksaUJBQVksR0FBWixZQUFZLENBQVE7UUFBVSw0QkFBdUIsR0FBdkIsdUJBQXVCLENBQU07UUFIakUscUJBQWdCLEdBQXdCLEVBQUUsQ0FBQztJQUtuRCxDQUFDO0lBYkQsa0JBQWtCO0lBQ2xCLE1BQU0sS0FBSyxTQUFTO1FBQ2xCLHlDQUF5QztRQUN6QywwRUFBMEU7UUFDMUUsb0RBQW9EO1FBQ3BELE9BQU8sU0FBUyxDQUFDO0lBQ25CLENBQUM7SUFTRCxjQUFjLENBQUMsaUJBQWlEO1FBQzlELE1BQU0sYUFB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LFlBQVksQ0FBQyxDQUFDO1FBQzFELE1BQU0sU0FBUyxHQUFHLEtBQUssQ0FBQztRQUN4QixJQUFJLENBQUMsZ0JBQWdCLEdBQUcsWUFBWSxDQUFDLEdBQUcsQ0FDcEMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUMsWUFBWSxFQUFFLENBQUMsQ0FBQyxJQUFJLEVBQUUsUUFBUSxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsUUFBUSxDQUFDLFNBQVMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdFLENBQUM7SUFFRCxTQUFTO1FBQ1AsT0FBTztZQUNMLGNBQWMsRUFBRSxJQUFJLENBQUMsWUFBWTtZQUNqQyx5QkFBeUIsRUFBRSxJQUFJLENBQUMsdUJBQXVCO1NBQ3hELENBQUM7SUFDSixDQUFDO0lBRUQsa0JBQWtCO0lBQ2xCLE1BQU0sQ0FBVSxVQUFVLENBQ3RCLEdBQStCLEVBQUUsTUFBa0I7UUFDckQsT0FBTyxJQUFJLEdBQUcsQ0FBQyxNQUFNLENBQUMsY0FBYyxDQUFDLEVBQUUsTUFBTSxDQUFDLHlCQUF5QixDQUFDLENBQUMsQ0FBQztJQUM1RSxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtkaXNwb3NlLCB0aWR5fSBmcm9tICcuLi9nbG9iYWxzJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9vcHMvYWRkJztcbmltcG9ydCB7ZGl2fSBmcm9tICcuLi9vcHMvZGl2JztcbmltcG9ydCB7ZmlsbH0gZnJvbSAnLi4vb3BzL2ZpbGwnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL29wcy9tdWwnO1xuaW1wb3J0IHtzcXJ0fSBmcm9tICcuLi9vcHMvc3FydCc7XG5pbXBvcnQge3NxdWFyZX0gZnJvbSAnLi4vb3BzL3NxdWFyZSc7XG5pbXBvcnQge0NvbmZpZ0RpY3QsIFNlcmlhbGl6YWJsZSwgU2VyaWFsaXphYmxlQ29uc3RydWN0b3J9IGZyb20gJy4uL3NlcmlhbGl6YXRpb24nO1xuaW1wb3J0IHtOYW1lZFRlbnNvciwgTmFtZWRWYXJpYWJsZU1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcblxuaW1wb3J0IHtPcHRpbWl6ZXIsIE9wdGltaXplclZhcmlhYmxlfSBmcm9tICcuL29wdGltaXplcic7XG5cbi8qKiBAZG9jbGluayBPcHRpbWl6ZXIgKi9cbmV4cG9ydCBjbGFzcyBBZGFncmFkT3B0aW1pemVyIGV4dGVuZHMgT3B0aW1pemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBnZXQgY2xhc3NOYW1lKCkge1xuICAgIC8vIE5hbWUgbWF0dGVycyBmb3IgUHl0aG9uIGNvbXBhdGliaWxpdHkuXG4gICAgLy8gVGhpcyBpcyBhIGdldHRlciBpbnN0ZWFkIG9mIGEgcHJvcGVydHkgYmVjYXVzZSB3aGVuIGl0J3MgYSBwcm9wZXJ0eSwgaXRcbiAgICAvLyBwcmV2ZW50cyB0aGUgZW50aXJlIGNsYXNzIGZyb20gYmVpbmcgdHJlZS1zaGFrZW4uXG4gICAgcmV0dXJuICdBZGFncmFkJztcbiAgfVxuXG4gIHByaXZhdGUgYWNjdW11bGF0ZWRHcmFkczogT3B0aW1pemVyVmFyaWFibGVbXSA9IFtdO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgcHJvdGVjdGVkIGxlYXJuaW5nUmF0ZTogbnVtYmVyLCBwcml2YXRlIGluaXRpYWxBY2N1bXVsYXRvclZhbHVlID0gMC4xKSB7XG4gICAgc3VwZXIoKTtcbiAgfVxuXG4gIGFwcGx5R3JhZGllbnRzKHZhcmlhYmxlR3JhZGllbnRzOiBOYW1lZFZhcmlhYmxlTWFwfE5hbWVkVGVuc29yW10pIHtcbiAgICBjb25zdCB2YXJpYWJsZU5hbWVzID0gQXJyYXkuaXNBcnJheSh2YXJpYWJsZUdyYWRpZW50cykgP1xuICAgICAgICB2YXJpYWJsZUdyYWRpZW50cy5tYXAoaXRlbSA9PiBpdGVtLm5hbWUpIDpcbiAgICAgICAgT2JqZWN0LmtleXModmFyaWFibGVHcmFkaWVudHMpO1xuXG4gICAgdmFyaWFibGVOYW1lcy5mb3JFYWNoKChuYW1lLCBpKSA9PiB7XG4gICAgICBjb25zdCB2YWx1ZSA9IEVOR0lORS5yZWdpc3RlcmVkVmFyaWFibGVzW25hbWVdO1xuICAgICAgaWYgKHRoaXMuYWNjdW11bGF0ZWRHcmFkc1tpXSA9PSBudWxsKSB7XG4gICAgICAgIGNvbnN0IHRyYWluYWJsZSA9IGZhbHNlO1xuICAgICAgICB0aGlzLmFjY3VtdWxhdGVkR3JhZHNbaV0gPSB7XG4gICAgICAgICAgb3JpZ2luYWxOYW1lOiBgJHtuYW1lfS9hY2N1bXVsYXRvcmAsXG4gICAgICAgICAgdmFyaWFibGU6IHRpZHkoXG4gICAgICAgICAgICAgICgpID0+IGZpbGwodmFsdWUuc2hhcGUsIHRoaXMuaW5pdGlhbEFjY3VtdWxhdG9yVmFsdWUpXG4gICAgICAgICAgICAgICAgICAgICAgICAudmFyaWFibGUodHJhaW5hYmxlKSlcbiAgICAgICAgfTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgZ3JhZGllbnQgPSBBcnJheS5pc0FycmF5KHZhcmlhYmxlR3JhZGllbnRzKSA/XG4gICAgICAgICAgdmFyaWFibGVHcmFkaWVudHNbaV0udGVuc29yIDpcbiAgICAgICAgICB2YXJpYWJsZUdyYWRpZW50c1tuYW1lXTtcbiAgICAgIGlmIChncmFkaWVudCA9PSBudWxsKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgY29uc3QgYWNjdW11bGF0ZWRHcmFkID0gdGhpcy5hY2N1bXVsYXRlZEdyYWRzW2ldLnZhcmlhYmxlO1xuXG4gICAgICB0aWR5KCgpID0+IHtcbiAgICAgICAgY29uc3QgbmV3QWNjdW11bGF0ZWRHcmFkID0gYWRkKGFjY3VtdWxhdGVkR3JhZCwgc3F1YXJlKGdyYWRpZW50KSk7XG4gICAgICAgIGFjY3VtdWxhdGVkR3JhZC5hc3NpZ24obmV3QWNjdW11bGF0ZWRHcmFkKTtcblxuICAgICAgICBjb25zdCBuZXdWYWx1ZSA9IGFkZChcbiAgICAgICAgICAgIG11bChkaXYoZ3JhZGllbnQsXG4gICAgICAgICAgICAgICAgICAgIHNxcnQoYWRkKG5ld0FjY3VtdWxhdGVkR3JhZCwgRU5HSU5FLmJhY2tlbmQuZXBzaWxvbigpKSkpLFxuICAgICAgICAgICAgICAgIC10aGlzLmxlYXJuaW5nUmF0ZSksXG4gICAgICAgICAgICB2YWx1ZSk7XG4gICAgICAgIHZhbHVlLmFzc2lnbihuZXdWYWx1ZSk7XG4gICAgICB9KTtcbiAgICB9KTtcbiAgICB0aGlzLmluY3JlbWVudEl0ZXJhdGlvbnMoKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgaWYgKHRoaXMuYWNjdW11bGF0ZWRHcmFkcyAhPSBudWxsKSB7XG4gICAgICBkaXNwb3NlKHRoaXMuYWNjdW11bGF0ZWRHcmFkcy5tYXAodiA9PiB2LnZhcmlhYmxlKSk7XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgYXN5bmMgZ2V0V2VpZ2h0cygpOiBQcm9taXNlPE5hbWVkVGVuc29yW10+IHtcbiAgICAvLyBPcmRlciBtYXR0ZXJzIGZvciBQeXRob24gY29tcGF0aWJpbGl0eS5cbiAgICByZXR1cm4gW2F3YWl0IHRoaXMuc2F2ZUl0ZXJhdGlvbnMoKV0uY29uY2F0KHRoaXMuYWNjdW11bGF0ZWRHcmFkcy5tYXAoXG4gICAgICAgIHYgPT4gKHtuYW1lOiB2Lm9yaWdpbmFsTmFtZSwgdGVuc29yOiB2LnZhcmlhYmxlfSkpKTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIHNldFdlaWdodHMod2VpZ2h0VmFsdWVzOiBOYW1lZFRlbnNvcltdKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgd2VpZ2h0VmFsdWVzID0gYXdhaXQgdGhpcy5leHRyYWN0SXRlcmF0aW9ucyh3ZWlnaHRWYWx1ZXMpO1xuICAgIGNvbnN0IHRyYWluYWJsZSA9IGZhbHNlO1xuICAgIHRoaXMuYWNjdW11bGF0ZWRHcmFkcyA9IHdlaWdodFZhbHVlcy5tYXAoXG4gICAgICAgIHYgPT4gKHtvcmlnaW5hbE5hbWU6IHYubmFtZSwgdmFyaWFibGU6IHYudGVuc29yLnZhcmlhYmxlKHRyYWluYWJsZSl9KSk7XG4gIH1cblxuICBnZXRDb25maWcoKTogQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHtcbiAgICAgICdsZWFybmluZ1JhdGUnOiB0aGlzLmxlYXJuaW5nUmF0ZSxcbiAgICAgICdpbml0aWFsQWNjdW11bGF0b3JWYWx1ZSc6IHRoaXMuaW5pdGlhbEFjY3VtdWxhdG9yVmFsdWUsXG4gICAgfTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgZnJvbUNvbmZpZzxUIGV4dGVuZHMgU2VyaWFsaXphYmxlPihcbiAgICAgIGNsczogU2VyaWFsaXphYmxlQ29uc3RydWN0b3I8VD4sIGNvbmZpZzogQ29uZmlnRGljdCk6IFQge1xuICAgIHJldHVybiBuZXcgY2xzKGNvbmZpZ1snbGVhcm5pbmdSYXRlJ10sIGNvbmZpZ1snaW5pdGlhbEFjY3VtdWxhdG9yVmFsdWUnXSk7XG4gIH1cbn1cbiJdfQ==