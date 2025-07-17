autumnstdlib = """
(module
  (= GRID_SIZE 16)
  (= FRAME_RATE 8)
  (= background "black")

  (= move (--> (obj dir) 
    (updateObj obj "origin" 
      (Position 
        (+ (.. (.. obj origin) x) (.. dir x))
        (+ (.. (.. obj origin) y) (.. dir y))
      )
    )
  ))
  
  (= moveRight (--> obj (move obj (Position 1 0))))
  (= moveLeft (--> obj (move obj (Position -1 0))))
  (= moveUp (--> obj (move obj (Position 0 -1))))
  (= moveDown (--> obj (move obj (Position 0 1))))

  (= moveNoCollision (--> (obj x y) 
    (if (isFreeExcept (move obj (Position x y)) obj) then (move obj (Position x y)) else obj)
  ))


  (= moveCanCollision (--> (obj x y obj2) 
    (if (isFreeExcept (move obj (Position x y)) (list obj obj2)) then (move obj (Position x y)) else obj)
  ))


  (= rotateNoCollision (--> (obj)
    (if (& (isWithinBounds (rotate obj)) (isFreeExcept (rotate obj) obj)) then (rotate obj) else obj)
  ))

  (= movePos (--> (pos dir)
    (Position 
      (+ (.. pos x) (.. dir x))
      (+ (.. pos y) (.. dir y))
    )
  ))

  (= moveRightPos (--> pos (movePos pos (Position 1 0))))
  (= moveLeftPos (--> pos (movePos pos (Position -1 0))))
  (= moveUpPos (--> pos (movePos pos (Position 0 -1))))
  (= moveDownPos (--> pos (movePos pos (Position 0 1))))

  (= abs (--> x (if (< x 0) then (- x) else x)))
  
  (= sign (--> x 
    (if (== x 0) 
      then 0 
      else (if (< x 0) then -1 else 1))
  ))

  (= vcat concat)
  
  (= deltaPos (--> (pos1 pos2) 
    (Position 
      (- (.. pos2 x) (.. pos1 x)) 
      (- (.. pos2 y) (.. pos1 y))
    )
  ))

  (= rect (--> (pos1 pos2) (
    let (= xmin (.. pos1 x))
        (= xmax (.. pos2 x))
        (= ymin (.. pos1 y))
        (= ymax (.. pos2 y))
        (= rectPos (concat (map (--> y (map (--> x (Position x y)) (range xmin xmax))) (range ymin ymax))))
        rectPos
  )))

  (= displacement deltaPos)

  (= deltaElem (--> (e1 e2)
    (deltaPos (.. e1 position) (.. e2 position))
  ))

  (= deltaObj (--> (obj1 obj2)
    (deltaPos (.. obj1 origin) (.. obj2 origin))
  ))


  (= adjacentElem (--> (e1 e2) (
      let (= delta (deltaPos (.. e1 position) (.. e2 position)))
          (== (+ (abs (.. delta x)) (abs (.. delta y))) 1)
      )
  ))

  (= adjacentPoss (--> (p1 p2 unitSize) (
    let (= delta (deltaPos p1 p2))
        (<= (+ (abs (.. delta x)) (abs (.. delta y))) unitSize)
    )
  ))

  (= adjacentTwoObjs (--> (obj1 obj2 unitSize) (
    adjacentPoss (.. obj1 origin) (.. obj2 origin) unitSize
  )))

  (= adjacentPossDiag (--> (p1 p2) (
    let (= delta (deltaPos p1 p2))
        (& (<= (abs (.. delta x)) 1)
           (<= (abs (.. delta y)) 1))

  )))

  (= adjacentTwoObjsDiag (--> (obj1 obj2) (
    adjacentPossDiag (.. obj1 origin) (.. obj2 origin)
  )))


  (= adjacentObjs (--> (obj1 unitSize) 
    (if (isList obj1) then 
    (filter (--> obj2 (any (--> o1 (adjacentTwoObjs o1 obj2 unitSize)) obj1)) ((allObjs)))
    else
    (filter (--> obj2 (adjacentTwoObjs obj1 obj2 unitSize)) ((allObjs)))
  )))

  (= adjacentObjsDiag (--> obj (
    if (isList obj) then 
    (filter (--> obj2 (any (--> o1 (adjacentTwoObjsDiag o1 obj2)) obj)) ((allObjs)))
    else
    (filter (--> obj2 (adjacentTwoObjsDiag obj obj2)) ((allObjs)))
  )))


  (= adj (--> (obj objs unitSize)
    (any (--> obj2 (adjacentTwoObjs obj obj2 unitSize)) objs)
  ))

  (= objClicked (--> (objs) (
    filter (--> obj (clicked obj)) objs
  )))

  (= sqdist (--> (pos1 pos2) 
           (let 
             (= delta (deltaPos pos1 pos2))
             (+ (* (.. delta x) (.. delta x)) (* (.. delta y) (.. delta y)))
           )
  ))
  
  (= unitVector (--> (obj target)
    (let
      (= delta (deltaPos (.. obj origin) (.. target origin)))
      (= sign_x (sign (.. delta x)))
      (= sign_y (sign (.. delta y)))
      (if (and (== (abs sign_x) 1)
               (== (abs sign_y) 1)) then
          (Position sign_x 0) else (Position sign_x sign_y)
      )
    )))

  (= unitVectorObjPos (--> (obj pos)
    (let
      (= delta (deltaPos (.. obj origin) pos))
      (= sign_x (sign (.. delta x)))
      (= sign_y (sign (.. delta y)))
      (if (and (== (abs sign_x) 1)
               (== (abs sign_y) 1)) then
          (Position sign_x 0) else (Position sign_x sign_y)
      )
    )
  ))

  (= unitVectorSinglePos (--> (pos)
    (let
      (= sign_x (sign (.. pos x)))
      (= sign_y (sign (.. pos y)))
      (Position sign_x sign_y)
    )
  ))

  (= max (--> (a b) (if (< a b) then b else a)))
  (= min (--> (a b) (if (< a b) then a else b)))

  (= closest (--> (obj listObjs) 
            (if (== (length listObjs) 0) then obj else
            (foldl (--> (obj1 obj2) 
              (if (< (sqdist (.. obj1 origin) (.. obj origin)) (sqdist (.. obj2 origin) (.. obj origin))) then obj1 else obj2
            )) (head listObjs) listObjs)
            )))

  (= closestPos (--> (obj listPoss)
            (if (== (length listPoss) 0) then (.. obj origin) else
            (foldl (--> (pos1 pos2) 
              (if (< (sqdist pos1 (.. obj origin)) (sqdist pos2 (.. obj origin))) then pos1 else pos2
            )) (head listPoss) listPoss)
            )))

  (= renderValue (--> obj (if (isList obj) then (concat (map renderValue obj)) else ((.. obj render)))))

  (= sum (--> (l) (foldl (--> (acc x) (+ acc x)) 0 l)))

  (= intersectsElems (--> (elems1 elems2) (
        if (or (== (length elems1) 0) (== (length elems2) 0)) then false else
        (any (--> elem1
                  (any (--> elem2 (== (.. elem1 position) (.. elem2 position))) elems2)) elems1)
        )
  ))

  (= intersectsPosElems (--> (pos elems) 
                          (any (--> elem (== (.. elem position) pos)) elems)))

  (= intersectsPosPoss (--> (pos poss) 
                          (any (--> pos2 (== pos pos2)) poss)))


  (= intersects (--> (obj1 obj2) (intersectsElems (renderValue obj1) (renderValue obj2))))

  (= isFree (--> obj (! (any (--> e (! (isFreePos (.. e position)))) (renderValue obj)))))


  (= in (--> (e l) (any (--> x (== x e)) l)))

  (= isFreeExcept (--> (obj prev_obj) (
    let (= prev_elems (renderValue prev_obj))
        (= curr_elems (renderValue obj))
        (= filtered_elems (filter (--> elem (! (in elem prev_elems))) curr_elems))
        (! (any (--> elem (! (isFreePos (.. elem position)))) filtered_elems))
  )))

  (= isFreePosExceptObj (--> (pos obj) (
    if (| (isFreePos pos) (intersectsPosElems pos (renderValue obj))) then true else false
    )
  ))

  (= isFreeRangeExceptObj (--> (start end obj) (
    let (= allCheckedPos (map (--> (x) (Position x 0)) (range start end)))
        (= prev_elems (renderValue obj))
        (= filtered_pos (filter (--> pos (! (intersectsPosElems pos prev_elems))) allCheckedPos))
        (! (any (--> pos (! (isFreePos pos))) filtered_pos))
  )))


  (= moveLeftNoCollision (--> obj (let 
      (if (and (isWithinBounds (moveLeft obj)) (isFreeExcept (moveLeft obj) obj)) then (moveLeft obj) else obj))))
  (= moveRightNoCollision (--> obj (let
      (= wbound (isWithinBounds (moveRight obj)))
      (= fr (isFreeExcept (moveRight obj) obj))
      (= ret (if (& wbound fr) then (moveRight obj) else obj))
      ret
      ))
  )
  (= moveUpNoCollision (--> obj (if (and (isWithinBounds (moveUp obj)) (isFreeExcept (moveUp obj) obj)) then (moveUp obj) else obj)))
  (= moveDownNoCollision (--> obj (if (and (isWithinBounds (moveDown obj)) (isFreeExcept (moveDown obj) obj)) then (moveDown obj) else obj)))

  (= nextSolid moveDownNoCollision)


  
  (= nextLiquidClosestHole (--> (obj holes) 
      (if (== (length holes) 0) then obj else (
              let (= closestHole (closestPos obj holes))
                  (nextLiquidMoveClosestHole obj closestHole)
          )
      ))
  )

  (= nextLiquid (--> (obj) (
    if (and
        (!= (.. (.. obj origin) y) (- GRID_SIZE 1))
        (isFree (moveDown obj))
        ) then (moveDown obj) else (
          let (= nextRowPos (rect (Position 0 (+ (.. (.. obj origin) y) 1)) (Position GRID_SIZE (+ (.. (.. obj origin) y) 2))))
              (= nextRowPos (filter (--> p (!= (.. p y) GRID_SIZE)) nextRowPos))
              (= holes (filter (--> p (and (isFreePos p) (isFreePos (moveUpPos p)))) nextRowPos))
              (nextLiquidClosestHole obj holes)
          )
    )))


  (= nextLiquidMoveClosestHole (--> (obj closestHole) (
     let (= dir (unitVectorObjPos obj (moveUpPos closestHole)))
         (= movedObj (move obj dir))
         (if (and (isFreePos (moveUpPos closestHole))
              (and (isFreePos (.. movedObj origin))
                    (isWithinBounds movedObj))
             ) then movedObj else obj)
     )
  ))

)
"""
