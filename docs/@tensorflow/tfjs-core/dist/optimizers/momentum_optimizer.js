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
import { mul } from '../ops/mul';
import { scalar } from '../ops/scalar';
import { zerosLike } from '../ops/zeros_like';
import { SGDOptimizer } from './sgd_optimizer';
/** @doclink Optimizer */
export class MomentumOptimizer extends SGDOptimizer {
    constructor(learningRate, momentum, useNesterov = false) {
        super(learningRate);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.useNesterov = useNesterov;
        this.accumulations = [];
        this.m = scalar(this.momentum);
    }
    /** @nocollapse */
    // Name matters for Python compatibility.
    static get className() {
        // Name matters for Python compatibility.
        // This is a getter instead of a property because when it's a property, it
        // prevents the entire class from being tree-shaken.
        return 'Momentum';
    }
    applyGradients(variableGradients) {
        const variableNames = Array.isArray(variableGradients) ?
            variableGradients.map(item => item.name) :
            Object.keys(variableGradients);
        variableNames.forEach((name, i) => {
            const value = ENGINE.registeredVariables[name];
            if (this.accumulations[i] == null) {
                const trainable = false;
                this.accumulations[i] = {
                    originalName: `${name}/momentum`,
                    variable: tidy(() => zerosLike(value).variable(trainable))
                };
            }
            const accumulation = this.accumulations[i].variable;
            const gradient = Array.isArray(variableGradients) ?
                variableGradients[i].tensor :
                variableGradients[name];
            if (gradient == null) {
                return;
            }
            tidy(() => {
                let newValue;
                const newAccumulation = add(mul(this.m, accumulation), gradient);
                if (this.useNesterov) {
                    newValue = add(mul(this.c, add(gradient, mul(newAccumulation, this.m))), value);
                }
                else {
                    newValue = add(mul(this.c, newAccumulation), value);
                }
                accumulation.assign(newAccumulation);
                value.assign(newValue);
            });
        });
        this.incrementIterations();
    }
    dispose() {
        this.m.dispose();
        if (this.accumulations != null) {
            dispose(this.accumulations.map(v => v.variable));
        }
    }
    /**
     * Sets the momentum of the optimizer.
     *
     * @param momentum
     */
    setMomentum(momentum) {
        this.momentum = momentum;
    }
    async getWeights() {
        // Order matters for Python compatibility.
        return [await this.saveIterations()].concat(this.accumulations.map(v => ({ name: v.originalName, tensor: v.variable })));
    }
    async setWeights(weightValues) {
        weightValues = await this.extractIterations(weightValues);
        const trainable = false;
        this.accumulations = weightValues.map(v => ({ originalName: v.name, variable: v.tensor.variable(trainable) }));
    }
    getConfig() {
        return {
            'learningRate': this.learningRate,
            'momentum': this.momentum,
            'useNesterov': this.useNesterov
        };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls(config['learningRate'], config['momentum'], config['useNesterov']);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibW9tZW50dW1fb3B0aW1pemVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHRpbWl6ZXJzL21vbWVudHVtX29wdGltaXplci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxPQUFPLEVBQUUsSUFBSSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ3pDLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDL0IsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUMvQixPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBQ3JDLE9BQU8sRUFBQyxTQUFTLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQU01QyxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFN0MseUJBQXlCO0FBQ3pCLE1BQU0sT0FBTyxpQkFBa0IsU0FBUSxZQUFZO0lBWWpELFlBQ3VCLFlBQW9CLEVBQVUsUUFBZ0IsRUFDekQsY0FBYyxLQUFLO1FBQzdCLEtBQUssQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUZDLGlCQUFZLEdBQVosWUFBWSxDQUFRO1FBQVUsYUFBUSxHQUFSLFFBQVEsQ0FBUTtRQUN6RCxnQkFBVyxHQUFYLFdBQVcsQ0FBUTtRQUp2QixrQkFBYSxHQUF3QixFQUFFLENBQUM7UUFNOUMsSUFBSSxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ2pDLENBQUM7SUFoQkQsa0JBQWtCO0lBQ2xCLHlDQUF5QztJQUN6QyxNQUFNLEtBQWMsU0FBUztRQUMzQix5Q0FBeUM7UUFDekMsMEVBQTBFO1FBQzFFLG9EQUFvRDtRQUNwRCxPQUFPLFVBQVUsQ0FBQztJQUNwQixD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QUUsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLEdBQUcsQ0FDOUQsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxZQUFZLEVBQUUsTUFBTSxFQUFFLENBQUMsQ0FBQyxRQUFRLEVBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMxRCxDQUFDO0lBRVEsS0FBSyxDQUFDLFVBQVUsQ0FBQyxZQUEyQjtRQUNuRCxZQUFZLEdBQUcsTUFBTSxJQUFJLENBQUMsaUJBQWlCLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDMUQsTUFBTSxTQUFTLEdBQUcsS0FBSyxDQUFDO1FBQ3hCLElBQUksQ0FBQyxhQUFhLEdBQUcsWUFBWSxDQUFDLEdBQUcsQ0FDakMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUMsWUFBWSxFQUFFLENBQUMsQ0FBQyxJQUFJLEVBQUUsUUFBUSxFQUFFLENBQUMsQ0FBQyxNQUFNLENBQUMsUUFBUSxDQUFDLFNBQVMsQ0FBQyxFQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdFLENBQUM7SUFFUSxTQUFTO1FBQ2hCLE9BQU87WUFDTCxjQUFjLEVBQUUsSUFBSSxDQUFDLFlBQVk7WUFDakMsVUFBVSxFQUFFLElBQUksQ0FBQyxRQUFRO1lBQ3pCLGFBQWEsRUFBRSxJQUFJLENBQUMsV0FBVztTQUNoQyxDQUFDO0lBQ0osQ0FBQztJQUVELGtCQUFrQjtJQUNsQixNQUFNLENBQVUsVUFBVSxDQUN0QixHQUErQixFQUFFLE1BQWtCO1FBQ3JELE9BQU8sSUFBSSxHQUFHLENBQ1YsTUFBTSxDQUFDLGNBQWMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxVQUFVLENBQUMsRUFBRSxNQUFNLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQztJQUN6RSxDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtkaXNwb3NlLCB0aWR5fSBmcm9tICcuLi9nbG9iYWxzJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9vcHMvYWRkJztcbmltcG9ydCB7bXVsfSBmcm9tICcuLi9vcHMvbXVsJztcbmltcG9ydCB7c2NhbGFyfSBmcm9tICcuLi9vcHMvc2NhbGFyJztcbmltcG9ydCB7emVyb3NMaWtlfSBmcm9tICcuLi9vcHMvemVyb3NfbGlrZSc7XG5pbXBvcnQge0NvbmZpZ0RpY3QsIFNlcmlhbGl6YWJsZSwgU2VyaWFsaXphYmxlQ29uc3RydWN0b3J9IGZyb20gJy4uL3NlcmlhbGl6YXRpb24nO1xuaW1wb3J0IHtTY2FsYXIsIFRlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3IsIE5hbWVkVmFyaWFibGVNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5cbmltcG9ydCB7T3B0aW1pemVyVmFyaWFibGV9IGZyb20gJy4vb3B0aW1pemVyJztcbmltcG9ydCB7U0dET3B0aW1pemVyfSBmcm9tICcuL3NnZF9vcHRpbWl6ZXInO1xuXG4vKiogQGRvY2xpbmsgT3B0aW1pemVyICovXG5leHBvcnQgY2xhc3MgTW9tZW50dW1PcHRpbWl6ZXIgZXh0ZW5kcyBTR0RPcHRpbWl6ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgLy8gTmFtZSBtYXR0ZXJzIGZvciBQeXRob24gY29tcGF0aWJpbGl0eS5cbiAgc3RhdGljIG92ZXJyaWRlIGdldCBjbGFzc05hbWUoKSB7XG4gICAgLy8gTmFtZSBtYXR0ZXJzIGZvciBQeXRob24gY29tcGF0aWJpbGl0eS5cbiAgICAvLyBUaGlzIGlzIGEgZ2V0dGVyIGluc3RlYWQgb2YgYSBwcm9wZXJ0eSBiZWNhdXNlIHdoZW4gaXQncyBhIHByb3BlcnR5LCBpdFxuICAgIC8vIHByZXZlbnRzIHRoZSBlbnRpcmUgY2xhc3MgZnJvbSBiZWluZyB0cmVlLXNoYWtlbi5cbiAgICByZXR1cm4gJ01vbWVudHVtJztcbiAgfVxuICBwcml2YXRlIG06IFNjYWxhcjtcbiAgcHJpdmF0ZSBhY2N1bXVsYXRpb25zOiBPcHRpbWl6ZXJWYXJpYWJsZVtdID0gW107XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBwcm90ZWN0ZWQgb3ZlcnJpZGUgbGVhcm5pbmdSYXRlOiBudW1iZXIsIHByaXZhdGUgbW9tZW50dW06IG51bWJlcixcbiAgICAgIHByaXZhdGUgdXNlTmVzdGVyb3YgPSBmYWxzZSkge1xuICAgIHN1cGVyKGxlYXJuaW5nUmF0ZSk7XG4gICAgdGhpcy5tID0gc2NhbGFyKHRoaXMubW9tZW50dW0pO1xuICB9XG5cbiAgb3ZlcnJpZGUgYXBwbHlHcmFkaWVudHModmFyaWFibGVHcmFkaWVudHM6IE5hbWVkVmFyaWFibGVNYXB8TmFtZWRUZW5zb3JbXSkge1xuICAgIGNvbnN0IHZhcmlhYmxlTmFtZXMgPSBBcnJheS5pc0FycmF5KHZhcmlhYmxlR3JhZGllbnRzKSA/XG4gICAgICAgIHZhcmlhYmxlR3JhZGllbnRzLm1hcChpdGVtID0+IGl0ZW0ubmFtZSkgOlxuICAgICAgICBPYmplY3Qua2V5cyh2YXJpYWJsZUdyYWRpZW50cyk7XG5cbiAgICB2YXJpYWJsZU5hbWVzLmZvckVhY2goKG5hbWUsIGkpID0+IHtcbiAgICAgIGNvbnN0IHZhbHVlID0gRU5HSU5FLnJlZ2lzdGVyZWRWYXJpYWJsZXNbbmFtZV07XG4gICAgICBpZiAodGhpcy5hY2N1bXVsYXRpb25zW2ldID09IG51bGwpIHtcbiAgICAgICAgY29uc3QgdHJhaW5hYmxlID0gZmFsc2U7XG4gICAgICAgIHRoaXMuYWNjdW11bGF0aW9uc1tpXSA9IHtcbiAgICAgICAgICBvcmlnaW5hbE5hbWU6IGAke25hbWV9L21vbWVudHVtYCxcbiAgICAgICAgICB2YXJpYWJsZTogdGlkeSgoKSA9PiB6ZXJvc0xpa2UodmFsdWUpLnZhcmlhYmxlKHRyYWluYWJsZSkpXG4gICAgICAgIH07XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IGFjY3VtdWxhdGlvbiA9IHRoaXMuYWNjdW11bGF0aW9uc1tpXS52YXJpYWJsZTtcbiAgICAgIGNvbnN0IGdyYWRpZW50ID0gQXJyYXkuaXNBcnJheSh2YXJpYWJsZUdyYWRpZW50cykgP1xuICAgICAgICAgIHZhcmlhYmxlR3JhZGllbnRzW2ldLnRlbnNvciA6XG4gICAgICAgICAgdmFyaWFibGVHcmFkaWVudHNbbmFtZV07XG4gICAgICBpZiAoZ3JhZGllbnQgPT0gbnVsbCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIHRpZHkoKCkgPT4ge1xuICAgICAgICBsZXQgbmV3VmFsdWU6IFRlbnNvcjtcbiAgICAgICAgY29uc3QgbmV3QWNjdW11bGF0aW9uID0gYWRkKG11bCh0aGlzLm0sIGFjY3VtdWxhdGlvbiksIGdyYWRpZW50KTtcbiAgICAgICAgaWYgKHRoaXMudXNlTmVzdGVyb3YpIHtcbiAgICAgICAgICBuZXdWYWx1ZSA9IGFkZChcbiAgICAgICAgICAgICAgbXVsKHRoaXMuYywgYWRkKGdyYWRpZW50LCBtdWwobmV3QWNjdW11bGF0aW9uLCB0aGlzLm0pKSksIHZhbHVlKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICBuZXdWYWx1ZSA9IGFkZChtdWwodGhpcy5jLCBuZXdBY2N1bXVsYXRpb24pLCB2YWx1ZSk7XG4gICAgICAgIH1cbiAgICAgICAgYWNjdW11bGF0aW9uLmFzc2lnbihuZXdBY2N1bXVsYXRpb24pO1xuICAgICAgICB2YWx1ZS5hc3NpZ24obmV3VmFsdWUpO1xuICAgICAgfSk7XG4gICAgfSk7XG4gICAgdGhpcy5pbmNyZW1lbnRJdGVyYXRpb25zKCk7XG4gIH1cblxuICBvdmVycmlkZSBkaXNwb3NlKCk6IHZvaWQge1xuICAgIHRoaXMubS5kaXNwb3NlKCk7XG4gICAgaWYgKHRoaXMuYWNjdW11bGF0aW9ucyAhPSBudWxsKSB7XG4gICAgICBkaXNwb3NlKHRoaXMuYWNjdW11bGF0aW9ucy5tYXAodiA9PiB2LnZhcmlhYmxlKSk7XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFNldHMgdGhlIG1vbWVudHVtIG9mIHRoZSBvcHRpbWl6ZXIuXG4gICAqXG4gICAqIEBwYXJhbSBtb21lbnR1bVxuICAgKi9cbiAgc2V0TW9tZW50dW0obW9tZW50dW06IG51bWJlcikge1xuICAgIHRoaXMubW9tZW50dW0gPSBtb21lbnR1bTtcbiAgfVxuXG4gIG92ZXJyaWRlIGFzeW5jIGdldFdlaWdodHMoKTogUHJvbWlzZTxOYW1lZFRlbnNvcltdPiB7XG4gICAgLy8gT3JkZXIgbWF0dGVycyBmb3IgUHl0aG9uIGNvbXBhdGliaWxpdHkuXG4gICAgcmV0dXJuIFthd2FpdCB0aGlzLnNhdmVJdGVyYXRpb25zKCldLmNvbmNhdCh0aGlzLmFjY3VtdWxhdGlvbnMubWFwKFxuICAgICAgICB2ID0+ICh7bmFtZTogdi5vcmlnaW5hbE5hbWUsIHRlbnNvcjogdi52YXJpYWJsZX0pKSk7XG4gIH1cblxuICBvdmVycmlkZSBhc3luYyBzZXRXZWlnaHRzKHdlaWdodFZhbHVlczogTmFtZWRUZW5zb3JbXSk6IFByb21pc2U8dm9pZD4ge1xuICAgIHdlaWdodFZhbHVlcyA9IGF3YWl0IHRoaXMuZXh0cmFjdEl0ZXJhdGlvbnMod2VpZ2h0VmFsdWVzKTtcbiAgICBjb25zdCB0cmFpbmFibGUgPSBmYWxzZTtcbiAgICB0aGlzLmFjY3VtdWxhdGlvbnMgPSB3ZWlnaHRWYWx1ZXMubWFwKFxuICAgICAgICB2ID0+ICh7b3JpZ2luYWxOYW1lOiB2Lm5hbWUsIHZhcmlhYmxlOiB2LnRlbnNvci52YXJpYWJsZSh0cmFpbmFibGUpfSkpO1xuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IENvbmZpZ0RpY3Qge1xuICAgIHJldHVybiB7XG4gICAgICAnbGVhcm5pbmdSYXRlJzogdGhpcy5sZWFybmluZ1JhdGUsXG4gICAgICAnbW9tZW50dW0nOiB0aGlzLm1vbWVudHVtLFxuICAgICAgJ3VzZU5lc3Rlcm92JzogdGhpcy51c2VOZXN0ZXJvdlxuICAgIH07XG4gIH1cblxuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIG92ZXJyaWRlIGZyb21Db25maWc8VCBleHRlbmRzIFNlcmlhbGl6YWJsZT4oXG4gICAgICBjbHM6IFNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LCBjb25maWc6IENvbmZpZ0RpY3QpOiBUIHtcbiAgICByZXR1cm4gbmV3IGNscyhcbiAgICAgICAgY29uZmlnWydsZWFybmluZ1JhdGUnXSwgY29uZmlnWydtb21lbnR1bSddLCBjb25maWdbJ3VzZU5lc3Rlcm92J10pO1xuICB9XG59XG4iXX0=