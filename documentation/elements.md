---
layout: documentation
title: API - elements
permalink: /api/elements/
---

These are used inside of [`Circuit`](https://vyrjana.github.io/pyimpspec/api/circuit) and [`Connection`](https://vyrjana.github.io/pyimpspec/api/connections) objects.
Check the page for the [base element class](https://vyrjana.github.io/pyimpspec/api/base-element) for information about the methods that are available for various elements.

**Table of Contents**

- [Capacitor](#pyimpspeccapacitor)
- [ConstantPhaseElement](#pyimpspecconstantphaseelement)
- [DeLevieFiniteLength](#pyimpspecdeleviefinitelength)
- [Gerischer](#pyimpspecgerischer)
- [HavriliakNegami](#pyimpspechavriliaknegami)
- [Inductor](#pyimpspecinductor)
- [Resistor](#pyimpspecresistor)
- [Warburg](#pyimpspecwarburg)
- [WarburgOpen](#pyimpspecwarburgopen)
- [WarburgShort](#pyimpspecwarburgshort)


### **pyimpspec.Capacitor**

Capacitor

    Symbol: C

    Z = 1/(j*2*pi*f*C)

    Variables
    ---------
    C: float = 1E-6 (F)

```python
class Capacitor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.ConstantPhaseElement**

Constant phase element

    Symbol: Q

    Z = 1/(Y*(j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1E-6 (F*s^(n-1))
    n: float = 0.95

```python
class ConstantPhaseElement(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.DeLevieFiniteLength**

de Levie pore (finite length)

    Symbol: Ls

    Z = (Ri*Rr)^(1/2)*coth(d*(Ri/Rr)^(1/2)*(1+Y*(2*pi*f*j)^n)^(1/2))/(1+Y*(2*pi*f*j)^n)^(1/2)

    Variables
    ---------
    Ri: float = 10.0 (ohm/cm)
    Rr: float = 1.0 (ohm*cm)
    Y: float = 0.01 (F*s^(n-1)/cm)
    n: float = 0.8
    d: float = 0.2 (cm)

```python
class DeLevieFiniteLength(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Gerischer**

Gerischer

    Symbol: G

    Z = 1/(Y*(k+j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S*s^n)
    k: float = 1.0 (s^-1)
    n: float = 0.5

```python
class Gerischer(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.HavriliakNegami**

Havriliak-Negami relaxation

    Symbol: H

    Z = (((1+(j*2*pi*f*t)^a)^b)/(j*2*pi*f*dC))

    Variables
    ---------
    dC: float = 1E-6 (F)
    t: float = 1.0 (s)
    a: float = 0.9
    b: float = 0.9

```python
class HavriliakNegami(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Inductor**

Inductor

    Symbol: L

    Z = j*2*pi*f*L

    Variables
    ---------
    L: float = 1E-6 (H)

```python
class Inductor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Resistor**

Resistor

    Symbol: R

    Z = R

    Variables
    ---------
    R: float = 1E+3 (ohm)

```python
class Resistor(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.Warburg**

Warburg (semi-infinite diffusion)

    Symbol: W

    Z = 1/(Y*(2*pi*f*j)^(1/2))

    Variables
    ---------
    Y: float = 1.0 (S*s^(1/2))

```python
class Warburg(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.WarburgOpen**

Warburg, finite space or open (finite length diffusion with reflective boundary)

    Symbol: Wo

    Z = coth((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S)
    B: float = 1.0 (s^n)
    n: float = 0.5

```python
class WarburgOpen(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`




### **pyimpspec.WarburgShort**

Warburg, finite length or short (finite length diffusion with transmissive boundary)

    Symbol: Ws

    Z = tanh((B*j*2*pi*f)^n)/((Y*j*2*pi*f)^n)

    Variables
    ---------
    Y: float = 1.0 (S)
    B: float = 1.0 (s^n)
    n: float = 0.5

```python
class WarburgShort(Element):
	kwargs
```

_Constructor parameters_

- `kwargs`



